import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from src.conv_mvm import quantized_conv


def _weights_init(m):
    classname = m.__class__.__name__
    # print(classname)
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)


class prune(nn.Module):
    def __init__(self, pruning_rate):
        super(prune, self).__init__()
        self.pruning_rate = pruning_rate
    
    def forward(self, x):
        new_nonzero = (x != 0).sum().item()
        self.sparsity = 1 - new_nonzero / x.numel()
        #print(f"  ori_sparcity {self.sparsity:.1%}")
        if self.pruning_rate > 0 and isinstance(x, torch.Tensor):
            k = int( self.pruning_rate * x.numel())
            if k > 0:
                threshold = torch.topk(torch.abs(x).view(-1), k, largest=False)[0][-1]
                mask = torch.abs(x) > threshold
                x = x * mask.float()

                new_nonzero = (x != 0).sum().item()
                self.sparsity = 1 - new_nonzero / x.numel()   
            
                #print(f" shape {x.shape} | goal_sparcity {self.pruning_rate:.1%} | ac_sparcity {self.sparsity:.1%}")
        return x


class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)

 
class BasicBlock_Quant(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, arch_args, stride=1):
        super(BasicBlock_Quant, self).__init__()
        self.experiment_state= arch_args.experiment_state
        self.prune  = prune (pruning_rate=arch_args.input_prune_rate) if self.experiment_state == "pruning" else lambda x: x
        
        self.conv1 = quantized_conv(in_planes, planes, arch_args, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        
        self.conv2 = quantized_conv(planes, planes, arch_args, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
      
         
        if stride != 1 or in_planes != planes:
            
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(planes)
                )           
            # self.shortcut = LambdaLayer(lambda x: F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, (planes )//4, (planes )//4),
            #                                           "constant", 0))
        
    def forward(self, input):   
        x = input[0]
        L0 = input[1]
        if self.experiment_state == "pruning":
            x = self.prune(x)
        out, L1 = self.conv1(x) 
        out = self.bn1(out)
        shortcut_out = self.shortcut(x) 
        out += shortcut_out
        out = x1 = F.leaky_relu(out)
        if self.experiment_state == "pruning":
            out = self.prune(out)
        out, L2 = self.conv2(out)
        out = self.bn2(out)
        out = out + x1
        out = F.leaky_relu(out)
    
        return [out, L0 + L1 + L2]


class ResNet(nn.Module):
    def __init__(self, num_blocks, in_channels, arch_args, start_chan, num_classes=10, block=BasicBlock_Quant):
        super(ResNet, self).__init__()
        self.experiment_state= arch_args.experiment_state
        self.prune  = prune (pruning_rate=arch_args.input_prune_rate) if self.experiment_state == "pruning" else lambda x: x
        self.in_planes = start_chan
        self.experiment_state= arch_args.experiment_state 
        self.conv1 = nn.Conv2d(in_channels, start_chan, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(start_chan)
        
        self.layer1 = self._make_layer(block, start_chan, num_blocks[0], arch_args, stride=1)
        self.layer2 = self._make_layer(block, start_chan * 2, num_blocks[1], arch_args, stride=2)
        self.layer3 = self._make_layer(block, start_chan * 4, num_blocks[2], arch_args, stride=2)
        self.layer4 = self._make_layer(block, start_chan * 8, num_blocks[2], arch_args, stride=2)
        self.bn2 = nn.BatchNorm1d(start_chan * 8)        
        self.classifier = nn.Linear(start_chan * 8, num_classes)
       

        self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, arch_args, stride):
        strides = [stride] + [1]*(num_blocks-1) # 1 1 1 2 1 1 2 1 1
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, arch_args, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.leaky_relu(out)
        [out, L1] = self.layer1([out, 0])
        [out, L2] = self.layer2([out, L1])
        [out, L3] = self.layer3([out, L2])  
        [out, L4] = self.layer4([out, L3])
        out = F.avg_pool2d(out, out.size()[3])  # change to max pool?
        out = out.view(out.size(0), -1)
        out = self.bn2(out)
        if self.experiment_state == "pruning":
            out = self.prune(out)
        out = self.classifier(out)
        return out, L4 
        """TODO: Create architecture argument for this loss term scalar"""




# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.nn.init as init
# from src.conv_mvm import quantized_conv
# import time

# from torch.utils.checkpoint import checkpoint

# def _weights_init(m):
#     classname = m.__class__.__name__
#     if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
#         init.kaiming_normal_(m.weight)

# class prune(nn.Module):
#     def __init__(self, pruning_rate):
#         super(prune, self).__init__()
#         self.pruning_rate = pruning_rate
    
#     def forward(self, x):
#         new_nonzero = (x != 0).sum().item()
#         self.sparsity = 1 - new_nonzero / x.numel()
#         if self.pruning_rate > 0 and isinstance(x, torch.Tensor):
#             k = int( self.pruning_rate * x.numel())
#             if k > 0:
#                 threshold = torch.topk(torch.abs(x).view(-1), k, largest=False)[0][-1]
#                 mask = torch.abs(x) > threshold
#                 x = x * mask.float()
#                 new_nonzero = (x != 0).sum().item()
#                 self.sparsity = 1 - new_nonzero / x.numel()   
#         return x

# class LambdaLayer(nn.Module):
#     def __init__(self, lambd):
#         super(LambdaLayer, self).__init__()
#         self.lambd = lambd

#     def forward(self, x):
#         return self.lambd(x)

# class BasicBlock_Quant(nn.Module):
#     expansion = 1

#     def __init__(self, in_planes, planes, arch_args, stride=1):
#         super(BasicBlock_Quant, self).__init__()
#         self.experiment_state= arch_args.experiment_state
#         self.prune  = prune (pruning_rate=arch_args.input_prune_rate) if self.experiment_state == "pruning" else lambda x: x
        
#         self.conv1 = quantized_conv(in_planes, planes, arch_args, kernel_size=3, stride=stride, padding=1, bias=False)
#         self.bn1 = nn.BatchNorm2d(planes)
#         self.conv2 = quantized_conv(planes, planes, arch_args, kernel_size=3, stride=1, padding=1, bias=False)
#         self.bn2 = nn.BatchNorm2d(planes)
#         self.shortcut = nn.Sequential()
        
#         if stride != 1 or in_planes != planes:           
#             self.shortcut = LambdaLayer(lambda x: F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, (planes )//4, (planes )//4),
#                                                       "constant", 0))
        
#     def forward(self, input):   
#         x = input[0]
#         L0 = input[1]  # 传入的损失(实际上是之前累积的VQ损失)
        
#         if self.experiment_state == "pruning":
#             x = self.prune(x)
            
#         start_time = time.time()
        
#         # 第一个卷积层 - 收集VQ损失
#         out, adc_loss1 = self.conv1(x)  
#         out = self.bn1(out)
#         shortcut_out = self.shortcut(x) 
#         out += shortcut_out
#         out = x1 = F.leaky_relu(out)
        
#         if self.experiment_state == "pruning":
#             out = self.prune(out)
            
#         # 第二个卷积层 - 收集VQ损失
#         out, adc_loss2 = self.conv2(out)  
#         out = self.bn1(out)
#         out = self.bn2(out)
#         out = out + x1
#         out = F.leaky_relu(out)
        
#         # 正确累积VQ损失
#         vq_loss1=0
#         vq_loss2=0
#         total_vq_loss = L0 + vq_loss1 + vq_loss2
#         total_ADC_loss = L0 + adc_loss1 + adc_loss2
#         # 添加调试输出
#         # print(f"DEBUG BasicBlock: vq_loss1={vq_loss1:.6f}, vq_loss2={vq_loss2:.6f}, " +
#         #     f"input_loss={L0:.6f}, total={total_vq_loss:.6f}")
        
#        # return [out, total_vq_loss,total_ADC_loss]  # 返回累积的VQ损失
#         return [out,  total_ADC_loss]  # 返回累积的VQ损失
    

# class ResNet(nn.Module):
#     def __init__(self, num_blocks, in_channels, arch_args, start_chan, num_classes=10, block=BasicBlock_Quant):
#         super(ResNet, self).__init__()
#         self.experiment_state= arch_args.experiment_state
#         self.prune  = prune (pruning_rate=arch_args.input_prune_rate) if self.experiment_state == "pruning" else lambda x: x
#         self.in_planes = start_chan
#         self.experiment_state= arch_args.experiment_state 
#         self.use_checkpointing = False   # 添加这个属性

        
#         # 第一个卷积层使用常规卷积
#         self.conv1 = nn.Conv2d(in_channels, start_chan, kernel_size=3, stride=1, padding=1, bias=False)
#         self.bn1 = nn.BatchNorm2d(start_chan)
        
#         self.layer1 = self._make_layer(block, start_chan, num_blocks[0], arch_args, stride=1)
#         self.layer2 = self._make_layer(block, start_chan * 2, num_blocks[1], arch_args, stride=2)
#         self.layer3 = self._make_layer(block, start_chan * 4, num_blocks[2], arch_args, stride=2)
#         self.layer4 = self._make_layer(block, start_chan * 8, num_blocks[2], arch_args, stride=2)
#         self.bn2 = nn.BatchNorm1d(start_chan * 8)        
#         self.classifier = nn.Linear(start_chan * 8, num_classes)

#         self.apply(_weights_init)

#     def _make_layer(self, block, planes, num_blocks, arch_args, stride):
#         strides = [stride] + [1]*(num_blocks-1)
#         layers = []
#         for stride in strides:
#             layers.append(block(self.in_planes, planes, arch_args, stride))
#             self.in_planes = planes * block.expansion

#         return nn.Sequential(*layers)
   
#     def forward(self, x):
#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = F.leaky_relu(out)
        
#         # 从0开始累积VQ损失
#         # [out, L1] =  checkpoint (self.layer1, [out, 0], use_reentrant=False )
#         # [out, L2] =  checkpoint (self.layer2, [out, L1], use_reentrant=False )
#         # [out, L3] =  checkpoint (self.layer3, [out, L2], use_reentrant=False )
#         # [out, L4] =  checkpoint (self.layer4, [out, L3], use_reentrant=False )
#         # [out, VQ1,ADC1] = self.layer1([out, 0,0])
#         # [out, VQ2,ADC2] = self.layer2([out, VQ1,ADC1])
#         # [out, VQ3,ADC3] = self.layer3([out, VQ2,ADC2])  
#         # [out, VQ4,ADC4] = self.layer4([out, VQ3,ADC3])
#         [out, ADC1] = self.layer1([out,0])
#         [out, ADC2] = self.layer2([out, ADC1])
#         [out, ADC3] = self.layer3([out, ADC2])  
#         [out, ADC4] = self.layer4([out, ADC3])
#         # 分类器
#         VQ4=0
#         out = F.avg_pool2d(out, out.size()[3])
#         out = out.view(out.size(0), -1)
#         out = self.bn2(out)
#         if self.experiment_state == "pruning":
#             out = self.prune(out)
#         out = self.classifier(out)
        
#         # 添加调试输出
#         # print(f"DEBUG ResNet: Layer losses: VQ1={VQ1:.6f}, VQ2={VQ2:.6f}, " +
#         #     f"VQ3={VQ3:.6f}, Final VQ4={VQ4:.6f}")
        
#         # 返回分类输出和总VQ损失
#         return out, VQ4 ,ADC4
       
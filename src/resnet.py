import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from src.conv_mvm import quantized_conv
from torch.utils.checkpoint import checkpoint

def _weights_init(m):
    classname = m.__class__.__name__
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)


class prune(nn.Module):
    def __init__(self, pruning_rate):
        super(prune, self).__init__()
        self.pruning_rate = pruning_rate
    
    def forward(self, x):
        if self.pruning_rate > 0 and isinstance(x, torch.Tensor):
            k = int(self.pruning_rate * x.numel())
            if k > 0:
                threshold = torch.topk(torch.abs(x).view(-1), k, largest=False)[0][-1]
                mask = torch.abs(x) > threshold
                x = x * mask.float()
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
        self.arch_args = arch_args  # 保存为实例变量
        self.experiment_state = arch_args.experiment_state
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
        
    def forward(self, input):   
        x = input[0]
        L0 = input[1]
      
        identity = x
        if self.experiment_state == "pruning":
            x = self.prune(x)
        # conv1
        out, L1 = self.conv1(x) 
        out = self.bn1(out)
        out = F.relu(out)  # ✅ 使用ReLU，不是LeakyReLU
        if self.experiment_state == "pruning":
            out = self.prune(out)
        # conv2
        out, L2 = self.conv2(out)
        out = self.bn2(out)
        
        # shortcut + relu
        out += self.shortcut(identity)  # ✅ 使用原始identity
        out = F.relu(out)  # ✅ 使用ReLU
    
        return [out, L0 + L1 + L2]


class ResNet(nn.Module):
    def __init__(self, num_blocks, in_channels, arch_args, start_chan, num_classes=10, block=BasicBlock_Quant):
        super(ResNet, self).__init__()
        self.arch_args = arch_args
        self.experiment_state = arch_args.experiment_state
        self.in_planes = start_chan
        self.prune  = prune (pruning_rate=arch_args.input_prune_rate) if self.experiment_state == "pruning" else lambda x: x
        
        self.conv1 = nn.Conv2d(in_channels, start_chan, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(start_chan)
        
        # 
        self.layer1 = self._make_layer(block, start_chan, num_blocks[0], arch_args, stride=1)
        self.layer2 = self._make_layer(block, start_chan * 2, num_blocks[1], arch_args, stride=2)
        self.layer3 = self._make_layer(block, start_chan * 4, num_blocks[2], arch_args, stride=2)
        
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(start_chan * 4, num_classes)
        
        self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, arch_args, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, arch_args, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        # 初始卷积
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out) 
        # 3个残差stage
        [out, L1] = self.layer1([out, 0])
        [out, L2] = self.layer2([out, L1])
        [out, L3] = self.layer3([out, L2])
        
        # 全局平均池化 + 分类
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        if self.experiment_state == "pruning":
            out = self.prune(out)
        out = self.classifier(out)
        
        return out, L3  # 

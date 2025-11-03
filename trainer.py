import os
import time
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import src.data_loaders as utilities
import dill
import src.argsparser as argsparser
from src.resnet import ResNet
import numpy as np
import torchviz
from tqdm import tqdm
from src.utils import Timer
from src.pruning import prune_model , apply_pruning
from src.adc_module import Nbit_ADC
import os
from torch.utils.checkpoint import checkpoint
from torch.amp import autocast, GradScaler 
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR, SequentialLR
from src.adc_module import gradientFilter
 






os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
os.environ['CUDA_VISIBLE_DEVICES'] = "1"
args = argsparser.get_parser().parse_args()
best_prec1 = 0

model_params = [args.model,
                args.dataset,
                args.epochs,
                args.batch_size,
                args.lr,
                args.experiment_state,
                args.weight_bits,
                args.bit_slice,
                args.wa_stoch_round,
                args.input_bits,
                args.bit_stream,
                args.save_adc,
                args.adc_bit,
                args.adc_grad_filter,
                args.adc_reg_lambda,
                # args.acm_fixed_bits,
                # args.acm_frac_bits,
                args.calibrate_adc,
                
                ]
quant_add = "_"
for item in model_params:
    quant_add += str(item).replace('.', 'p') + '_'
quant_add = quant_add[:-1]  # Remove the last underscore

run_name = args.run_info + quant_add
print(f"Run info: {args.run_info}")
print(f"Run name: {run_name }")

model_save_dir = args.model_save_dir + run_name + ".th"
logs_save_dir = args.logs_save_dir + run_name + ".txt"

if not os.path.exists("./saved/"):
    os.mkdir("./saved/")

if not os.path.exists(args.model_save_dir):
    os.mkdir(args.model_save_dir)

if not os.path.exists(args.logs_save_dir):
    os.mkdir(args.logs_save_dir)

if not os.path.exists("./saved/hist_csvs/"):
    os.mkdir("./saved/hist_csvs/")

# if not args.evaluate:
Log_Vals = open(logs_save_dir, 'w')

def setup_adc_names(model):
    """为所有ADC模块设置唯一名称"""
    adc_count = 0
    for name, module in model.named_modules():
        if hasattr(module, 'adc_pos'):
            module.adc_pos.adc_name = f"{name}.adc_pos"
            adc_count += 1
        if hasattr(module, 'adc_neg'):
            module.adc_neg.adc_name = f"{name}.adc_neg" 
            adc_count += 1
    
    print(f"设置了 {adc_count} 个ADC的名称")
    return adc_count
def main():
   
    start_time = time.time()
    global args, best_prec1
    print(f"Time @ args: {time.time() - start_time}")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cuda':
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        print("✓ CUDNN optimizations enabled")
    # Build model according to params
    if args.model == "resnet20":
        num_blocks = [3, 3, 3]
        start_chan = 16
    elif args.model == "resnet18":
        num_blocks = [2, 2, 2, 2]
        start_chan = 64
    else:
        raise NotImplementedError("Not a valid model for current codebase")

    if args.dataset == 'MNIST':
        model = ResNet(num_blocks, 1, args, start_chan).to(device)
    elif args.dataset == 'CIFAR10':
        model = ResNet(num_blocks, 3, args, start_chan).to(device)
    else:
        raise NotImplementedError("Not a valid dataset for current codebase")

    for name, module in model.named_modules():
        if module.__class__.__name__ == 'quantized_conv':
            module.layer_name = name
 
    print(f"Time @ model load: {time.time()-start_time}")

    # optionally resume from a checkpoint
    
    setup_adc_names(model)
    if args.resume:  # Model file must match arch, good luck
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, weights_only=False)
            
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'],strict=False)#
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
            
            adc_loaded_correctly = False
            for name, module in model.named_modules():
                if hasattr(module, 'adc_pos') and hasattr(module.adc_pos, 'step_size'):
                    step_size = module.adc_pos.step_size.item()
                    if abs(step_size - 1.0) > 1e-6:  # 不是默认值1.0
                        adc_loaded_correctly = True
                        print(f"  {name}.adc_pos: step_size={step_size:.6f}")
                        break
            
            if adc_loaded_correctly:
                print("✓ ADC校准参数已从checkpoint自动恢复")
            else:
                print("⚠ ADC参数为默认值，可能需要重新校准")
            #==============================================================
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    
    cudnn.benchmark = False
    train_loader, val_loader = utilities.get_loaders(dataset=args.dataset, batch_size=args.batch_size, workers=4)
    print(f"Time @ data load: {time.time()-start_time}")

    criterion = nn.CrossEntropyLoss().cuda()



   #====================================================================================================
    #                           ADC trainable factor
   #====================================================================================================
    # ✅ 分组参数：ADC 用更小的学习率
      # ✅ 确认：打印所有可训练参数
    # print("\n所有可训练参数:")
    # total_params = 0
    # step_params = 0
    
    # for name, param in model.named_parameters():
    #     if param.requires_grad:
    #         total_params += 1
    #         if 'step_size' in name:
    #             step_params += 1
    #             print(f"  ✓ {name}: requires_grad={param.requires_grad}")
    
    # print(f"\n总可训练参数: {total_params}")
    # print(f"Step_size 参数: {step_params}")
    
    # if step_params == 0:
    #     print("❌ 错误：没有找到可训练的 step_size 参数！")
    #     exit(1)
    
    # ✅ 分组优化器
    # adc_params = []
    # other_params = []
    
    # for name, param in model.named_parameters():
    #     if 'step_size' in name:
    #         adc_params.append(param)
    #     else:
    #         other_params.append(param)
    
    # optimizer = torch.optim.SGD([
    #     {'params': other_params, 'lr': args.lr},
    #     {'params': adc_params, 'lr': args.lr * 0.1}
    # ], momentum=0.9, weight_decay=args.weight_decay)
    
    # print(f"优化器设置: Main LR={args.lr}, ADC LR={args.lr * 0.1}")
#============================================ end ===============================================


     #optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    #optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    optimizer =  torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
   
    # #lr_scheduler =  torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[80, 120], gamma=0.1)

    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min=1e-6, last_epoch=-1)
 
     
    start_train = time.time()
    
    print(f"Time @ start: {time.time()-start_time}")
    

 
    if (len(args.resume) > 5) and args.save_adc:        
        validate(val_loader, model, criterion)
        exit()

    
    if (len(args.resume) > 5) and args.experiment_state == "inference":
        model.eval() 
        with torch.no_grad():
            validate(val_loader, model, criterion)
        exit()
    

    # ==========================================
    # new: PTQAT calibration step
    # This step is only executed if args.experiment_state is "PTQAT" and
    # args.enable_calibration is True.
    # ==========================================
    if args.experiment_state == "PTQAT" and args.calibrate_adc:
        print("=" * 60)
        print("开始 Array-wise ADC 校准...")
        print("=" * 60)
        model.eval()
        
        # ✅ 收集所有 ADC
        adc_list = []
        for module in model.modules():
            if hasattr(module, 'adc_pos_list'):
                adc_list.extend(module.adc_pos_list)
            if hasattr(module, 'adc_neg_list'):
                adc_list.extend(module.adc_neg_list)
        
        print(f"找到 {len(adc_list)} 个 ADC 需要校准")
        
        # ✅ 启动校准
        for adc in adc_list:
            adc.start_calibration()
        
        # ✅ 前向传播收集数据
        print("收集校准数据...")
        with torch.no_grad():
            for i, (input, target) in enumerate(train_loader):
                if i >= 20:  # 20个batch足够
                    break
                input_var = input.cuda()
                output, _ = model(input_var)
                if (i + 1) % 5 == 0:
                    print(f"  进度: {i + 1}/20")
        
        # ✅ 完成校准
        print("计算校准参数...")
        for adc in adc_list:
            adc.finish_calibration()
        
        # ✅ 简单验证
        step_sizes = [adc.step_size.item() for adc in adc_list]
        print(f"校准完成! step_size 范围: [{min(step_sizes):.6f}, {max(step_sizes):.6f}]")
        print(f"唯一值数量: {len(set([round(s, 6) for s in step_sizes]))}/{len(step_sizes)}")
        print("=" * 60)
    # ==========================================
    # end of PTQAT calibration step
    # ==========================================
    

    # ==========================================
    


    # begin epoch training loop
    for epoch in range(0, args.epochs):
        start_epoch = time.time()
        if args.experiment_state == "pruning":
            prune_model(model, args.conv_prune_rate, args.linear_prune_rate)
        print('current lr {:.5e}'.format(optimizer.param_groups[0]['lr']))

        train(train_loader, model, criterion, optimizer, epoch )
        lr_scheduler.step() 

        # evaluate on validation set
        prec1 = validate(val_loader, model, criterion)
         # ✅ 每5个epoch打印ADC参数
        # if (epoch + 1) % 5 == 0:
        #     print_adc_parameters(model)
        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        # Remove pruning reparameterization before saving
        if args.experiment_state == "pruning":
            model = apply_pruning(model)  # <-- Apply before saving
        #if is_best:
        save_checkpoint({
                'model': model,
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_prec1': best_prec1,
                'args': args
            }, filename=os.path.join(model_save_dir))
        print("Epoch Time: " + str(time.time() - start_epoch))
        Log_Vals.write(str(prec1) + '\n')
    print("Total Time: " + str(time.time() - start_train))






def train(train_loader, model, criterion, optimizer, epoch):
    """Run one train epoch"""
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    
    model.train()
    end = time.time()
   
    
    for i, (input, target) in enumerate(train_loader):
       
        # if i == 1:
        #     print("\n=== Gradient Statistics (after 1st batch) ===")
        #     for name, param in model.named_parameters():
        #         if param.grad is not None and 'weight' in name:
        #             print(f"{name}: grad mean={param.grad.mean():.6f}, "
        #                 f"std={param.grad.std():.6f}, "
        #                 f"max={param.grad.abs().max():.6f}")
        if i % 10 == 0:
            torch.cuda.empty_cache()
        data_time.update(time.time() - end)

        target = target.cuda()
        input_var = input.cuda()
        target_var = target.cuda()

        # compute output
        output, adc_loss = model(input_var)
        
        loss = criterion(output, target_var) 
         
     
        optimizer.zero_grad()
        loss.backward()
        if i % 100 == 0:
            for name, param in model.named_parameters():
                if 'alpha' in name and param.grad is not None:
                    print(f"{name}: value={param.data.item():.4f}, "
                          f"grad={param.grad.item():.4f}")
        optimizer.step() 
        
        loss = loss.float()
         
        
        # measure accuracy and record loss
        prec1 = accuracy(output.data, target)[0]
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))

        batch_time.update(time.time() - end)
        end = time.time()
        
        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                      epoch+1, i+1, len(train_loader), batch_time=batch_time,
                      data_time=data_time, loss=losses, top1=top1))
    Log_Vals.write(str(epoch+1) + ', ' + str(losses.avg) + ', ' + str(top1.avg) + ', ')
    
    
def validate(val_loader, model, criterion):
    """Run evaluation"""
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # ✅ 关键：必须设置为eval模式
    model.eval()
    
    end = time.time()
    
    with torch.no_grad():  # ✅ 整个验证过程都不需要梯度
        for i, (input, target) in enumerate(val_loader):
            last_finished_batch = args.skip_to_batch
            if i < last_finished_batch:
                continue

            target = target.cuda()
            input_var = input.cuda()

            # compute output
            output, adc_loss = model(input_var)
            
            # ✅ 验证时只用交叉熵
            loss = criterion(output, target)

            # measure accuracy and record loss
            prec1 = accuracy(output.data, target)[0]
            losses.update(loss.item(), input.size(0))
            top1.update(prec1.item(), input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            
            if args.print_batch_info:
                string = f"Batch {i+1} Prec@1 = {prec1:.2f}%, Avg = {top1.avg:.2f}%"
                print(string)

    print(' * Prec@1 {top1.avg:.3f}'.format(top1=top1))
    return top1.avg


def save_checkpoint(state, filename='checkpoint.pth.tar', is_best=False):
    """Save the training model"""
    torch.save(state, filename, pickle_module=dill)
    if is_best:
        best_filename = filename.replace('.th', '_best.th')
        torch.save(state, best_filename, pickle_module=dill)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

from collections import OrderedDict

def adjust_state_dict(state_dict):
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if 'conv' in k and not k.endswith('.1.weight') and not k.endswith('.1.bias'):
            k = k.replace('.weight', '.1.weight').replace('.bias', '.1.bias')
        new_state_dict[k] = v
    return new_state_dict
def print_adc_parameters(model):
    """打印所有ADC的可训练参数"""
    print("\n" + "="*70)
    print("ADC Trainable Parameters:")
    print("="*70)
    
    for name, module in model.named_modules():
        if hasattr(module, 'adc_pos'):
            adc = module.adc_pos
            if hasattr(adc, 'alpha'):
                alpha = adc.alpha.detach()
                print(f"{name}.adc_pos (1-bit):")
                print(f"  α: mean={alpha.mean():.4f}, std={alpha.std():.4f}, "
                      f"min={alpha.min():.4f}, max={alpha.max():.4f}")
            elif hasattr(adc, 'step_size'):
                step = adc.step_size.detach()
                print(f"{name}.adc_pos ({adc.bits}-bit):")
                print(f"  step_size: mean={step.mean():.4f}, std={step.std():.4f}, "
                      f"min={step.min():.4f}, max={step.max():.4f}")
        
        if hasattr(module, 'adc_neg'):
            adc = module.adc_neg
            if hasattr(adc, 'alpha'):
                alpha = adc.alpha.detach()
                print(f"{name}.adc_neg (1-bit):")
                print(f"  α: mean={alpha.mean():.4f}, std={alpha.std():.4f}, "
                      f"min={alpha.min():.4f}, max={alpha.max():.4f}")
            elif hasattr(adc, 'step_size'):
                step = adc.step_size.detach()
                print(f"{name}.adc_neg ({adc.bits}-bit):")
                print(f"  step_size: mean={step.mean():.4f}, std={step.std():.4f}, "
                      f"min={step.min():.4f}, max={step.max():.4f}")
    
    print("="*70 + "\n")
def setup_adc_names(model):
    """为所有 ADC 模块设置唯一名称（Array-wise）"""
    adc_count = 0
    for layer_name, module in model.named_modules():
        if hasattr(module, 'adc_pos_list'):
            for idx, adc in enumerate(module.adc_pos_list):
                adc.adc_name = f"{layer_name}.array{idx}.adc_pos"
                adc_count += 1
        if hasattr(module, 'adc_neg_list'):
            for idx, adc in enumerate(module.adc_neg_list):
                adc.adc_name = f"{layer_name}.array{idx}.adc_neg"
                adc_count += 1
    
    print(f"设置了 {adc_count} 个 ADC 的名称（Array-wise）")
    return adc_count
if __name__ == '__main__':
    main()

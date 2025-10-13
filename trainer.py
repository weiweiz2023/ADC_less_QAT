import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'

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
os.environ['CUDA_VISIBLE_DEVICES'] = "3"
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
            # 验证ADC参数是否正确加载=================================
            print("验证ADC参数加载状态:")
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
    #optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
   
   
    
    # ========== 新的 scheduler 配置 ==========
   
    # warmup_epochs = 5
    # total_epochs = args.epochs
    # def lr_lambda(epoch):
    #     if epoch < warmup_epochs:
    #         # Warmup: 从 0.1 线性增加到 1.0
    #         return 0.1 + 0.9 * (epoch / warmup_epochs)
    #     else:
    #         # Cosine annealing: 从 1.0 衰减到 eta_min_ratio
    #         progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
    #         eta_min_ratio = 1e-6 / args.lr  # 最小LR占初始LR的比例
    #         return eta_min_ratio + (1.0 - eta_min_ratio) * 0.5 * (1 + math.cos(math.pi * progress))

    # lr_scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)
    # ========== end==========
    
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
# VQ Calibration Step (before training)
# ==========================================
    if args.experiment_state == "PTQAT" and args.use_vq:
        print("=" * 60)
        print("开始VQ码本校准（K-means初始化）")
        print("=" * 60)
        
        model.eval()
        
        for name, module in model.named_modules():
            if hasattr(module, 'vq_layer') and module.vq_layer is not None:
                print(f"校准层: {name}")
                
                # 获取该层权重
                weights = module.weight.data.flatten()
                
                # 采样权重
                num_samples = min(10000, len(weights))
                indices = torch.randperm(len(weights))[:num_samples]
                samples = weights[indices].unsqueeze(1)
                
                # 用分位数初始化码本
                sorted_weights = torch.sort(samples.squeeze())[0]
                num_codes = module.vq_layer.num_embeddings
                quantile_indices = torch.linspace(0, len(sorted_weights)-1, num_codes).long()
                centroids = sorted_weights[quantile_indices].unsqueeze(1)
                
                # 运行5轮K-means
                for _ in range(10):
                    # 分配到最近中心
                    distances = torch.cdist(samples, centroids)
                    assignments = torch.argmin(distances, dim=1)
                    
                    # 更新中心
                    new_centroids = []
                    for k in range(num_codes):
                        mask = assignments == k
                        if mask.sum() > 0:
                            new_centroids.append(samples[mask].mean(dim=0, keepdim=True))
                        else:
                            new_centroids.append(centroids[k:k+1])
                    centroids = torch.cat(new_centroids, dim=0)
                
                # 更新VQ码本
                module.vq_layer.embedding.weight.data = centroids
                
                print(f"  码本范围: [{centroids.min():.4f}, {centroids.max():.4f}]")
        
        print("VQ码本校准完成！")
# ==========================================
    # end of VQ calibration step
    # ==========================================
    # ==========================================
    # new: PTQAT calibration step
    # This step is only executed if args.experiment_state is "PTQAT" and
    # args.enable_calibration is True.
    # ==========================================
        
    if args.experiment_state == "PTQAT" and args.calibrate_adc:
        print("=" * 60)
        print("开始ADC校准...")
        print("=" * 60)
        model.eval()  
        
        # 收集所有ADC模块
        adc_modules = []
        adc_count = 0
        
        for name, module in model.named_modules():
            if hasattr(module, 'adc_pos'):
                module.adc_pos.adc_name = f"{name}.adc_pos"
                adc_modules.append((f"{name}.adc_pos", module.adc_pos))
                adc_count += 1
            if hasattr(module, 'adc_neg'):
                module.adc_neg.adc_name = f"{name}.adc_neg" 
                adc_modules.append((f"{name}.adc_neg", module.adc_neg))
                adc_count += 1
        
        print(f"找到 {adc_count} 个ADC模块需要校准")
        
        # 步骤1：启动所有ADC的校准模式
        print("\n步骤1: 启动校准模式...")
        for name, adc in adc_modules:
            adc.start_calibration()
        
        # 步骤2：通过前向传播让每个ADC收集真实数据
        print("\n步骤2: 通过前向传播收集校准数据...")
        calibration_batches =20
        
        with torch.no_grad():
            for i, (input, target) in enumerate(train_loader):
                if i >= calibration_batches:
                    break
                input_var = input.cuda()
                try:
                    output, _ = model(input_var)
                    if (i + 1) % 5 == 0:
                        print(f"  进度: {i + 1}/{calibration_batches} 批次")
                except Exception as e:
                    print(f"  错误在批次 {i}: {e}")
                    continue
        
        # 步骤3：处理收集到的数据并计算校准参数
        print("\n步骤3: 计算校准参数...")
        step_sizes = []
        
        for i, (name, adc) in enumerate(adc_modules):
            print(f"\n[{i+1}/{len(adc_modules)}] 处理 {name}")
            adc.finish_calibration()
            
            step_size = adc.step_size.item() if hasattr(adc.step_size, 'item') else adc.step_size
            step_sizes.append(step_size)
        
        # 步骤4：验证校准结果
        print("\n" + "=" * 60)
        print("校准完成! ADC参数总结:")
        print("=" * 60)
        
        for name, step_size in zip([name for name, _ in adc_modules], step_sizes):
            print(f"  {name}: step_size = {step_size:.6f}")
        
        # 检查step_size的多样性
        unique_step_sizes = len(set([round(s, 6) for s in step_sizes]))
        print(f"\n发现 {unique_step_sizes} 种不同的step_size值")
        
        if unique_step_sizes <= 2:
            print("⚠️  警告: ADC参数缺乏多样性，可能存在校准问题")
            print("建议检查:")
            print("- 模型是否在正确的实验状态")
            print("- 前向传播是否正常执行")
            print("- ADC是否在正确的位置收集数据")
        else:
            print("✓ ADC参数显示良好的多样性，校准看起来正常")
        
        print("=" * 60)
    # ==========================================
    # end of PTQAT calibration step
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








def train(train_loader, model, criterion, optimizer, epoch ):
    """
        Run one train epoch
    """
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    vq_losses = AverageMeter()
    adc_losses = AverageMeter()  
     
     
    # 🆕 添加这里 - 动态VQ损失权重
    if args.use_vq:
        # 指数衰减：epoch 0→1, epoch 5→0.5, epoch 10→0.25
        vq_weight = args.vq_loss_weight * (0.5 ** (epoch / 5))
        print(f'VQ Loss Weight: {vq_weight:.6f} (原始: {args.vq_loss_weight})')
    else:
        vq_weight = 0.0
    # 🆕 添加结束

    # switch to train mode
    model.train()

    end = time.time()

    
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        if i % 10 == 0:
            torch.cuda.empty_cache()
        data_time.update(time.time() - end)

        target = target.cuda()
        input_var = torch.autograd.Variable(input.cuda())
        target_var = torch.autograd.Variable(target.cuda())

        # compute output
        start_forward = time.time() 
        output,vq_loss = model(input_var)
        #print("vq_loss",vq_loss) 
        start_backward = time.time()
        for module in model.modules():
            if hasattr(module, 'last_vq_loss'):
                vq_loss += module.last_vq_loss
        
        vq_losses.update(vq_loss.item(), input.size(0))
        loss = criterion(output, target_var)+  vq_weight *  vq_loss 
        
       
     
        optimizer.zero_grad()
        # scaler.scale(loss).backward()
        # scaler.step(optimizer)
        # scaler.update() 

        loss.backward()
        optimizer.step() 
        
        
        loss = loss.float()
        backward_time = time.time() - start_backward
        # measure accuracy and record loss
        prec1 = accuracy(output.data, target)[0]
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        # if i % 50 == 0:
        #     print(f"Forward: {forward_time:.3f}s, Backward: {backward_time:.3f}s")
        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  f'VQ Loss {vq_losses.val:.6f} ({vq_losses.avg:.6f})\t'  # 🆕 添加这行
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                      epoch+1, i+1, len(train_loader), batch_time=batch_time,
                      data_time=data_time, loss=losses, top1=top1))
    Log_Vals.write(str(epoch+1) + ', ' + str(losses.avg) + ', ' + str(top1.avg) + ', ')


def validate(val_loader, model, criterion):
    """
    Run evaluation
    """
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        last_finished_batch = args.skip_to_batch
        if i < last_finished_batch:
            continue

        target = target.cuda()
        with torch.no_grad():
            input_var = torch.autograd.Variable(input.cuda())
            target_var = torch.autograd.Variable(target.cuda())

        # compute output
           # output, vq_loss,adc_loss  = model(input_var)
            output ,vq_loss  = model(input_var)
            loss = criterion(output, target_var)  +   vq_loss  

            output = output.float()
            loss = loss.float()

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
            with open(args.batch_log, 'a') as log_file:
                log_file.write(string)


    print(' * Prec@1 {top1.avg:.3f}'
          .format(top1=top1))
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
if __name__ == '__main__':
    main()

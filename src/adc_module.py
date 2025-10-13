import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
import numpy as np
import os
import random

class Nbit_ADC(nn.Module):
    def __init__(self, bits: int, n_state_slices: int, n_state_stream: int, save_adc: bool,
                 adc_grad_filter: bool, custom_loss: bool = False, regularization_lambda: float = 0.01):
        super(Nbit_ADC, self).__init__()
        self.bits = bits
        self.save = save_adc
        self.grad_filter = adc_grad_filter 
        self.custom_loss = custom_loss  # 添加缺失的属性
        self.reg_lambda = regularization_lambda  # 正则化系数
        n_slices = n_state_slices
        n_stream = n_state_stream
        self.calibrating = False
        self.cal_data = []
        
        # Sampling parameters
        self.max_save_points = 10000  # Maximum points to save per ADC
        self.saved_points = 0
        self.sampling_probability = 1.0  # Dynamic sampling probability
        self.collection_completed = False   
        
        # Use register_buffer for state parameters
         #self.step_size = nn.Parameter(torch.tensor(1.0))
        self.register_buffer('step_size', torch.tensor(1.0))
        self.register_buffer('zero_point', torch.tensor(0.0))
        self.register_buffer('min_val_base', torch.tensor(0.0))
        self.register_buffer('max_val_base', torch.tensor(2 ** (bits - 1)))
    def start_calibration(self):
        """Start calibration mode"""
        print(f"ADC {getattr(self, 'adc_name', 'unknown')} starting calibration...")
        self.calibrating = True
        self.cal_data = []
    
    def finish_calibration(self):
        """Complete calibration and compute parameters"""
        if not self.cal_data:
            print(f"  Warning: ADC {getattr(self, 'adc_name', 'unknown')} collected no calibration data")
            self.calibrating = False
            return
            
        all_data = torch.cat(self.cal_data)
        total_points = all_data.numel()
        print(f"  Collected {len(self.cal_data)} batches, {total_points} data points")
        print(f"  Data range: [{all_data.min():.4f}, {all_data.max():.4f}]")
        print(f"  Mean: {all_data.mean():.4f}, Std: {all_data.std():.4f}")
        
        # Sample for quantile calculation if too many points
        if total_points >2000:
            indices = torch.randperm(total_points)[:2000]
            sample_data = all_data[indices]
        else:
            sample_data = all_data

        data_range = torch.quantile(sample_data, 0.95)- torch.quantile(sample_data, 0.05)
        
        
        step_size = data_range / (2**(self.bits-1))
        step_size_clamped = torch.clamp(step_size, 1e-6, 1e2)
        
        self.step_size.data.copy_(step_size_clamped)
        self.zero_point.data.copy_(torch.tensor(0.0))
        print(f"  Computed step_size: {self.step_size:.6f}")
        
        self.calibrating = False
        self.cal_data = []

    def _should_save_data(self, data_size):
        """Determine if data should be saved based on current state"""
        if not self.save or self.saved_points >= self.max_save_points:
            return False
        
        # Adaptive sampling: reduce probability as we collect more data
        remaining_capacity = self.max_save_points - self.saved_points
        if data_size > remaining_capacity:
            self.sampling_probability = remaining_capacity / data_size
        
        return random.random() < self.sampling_probability

    def _sample_and_save_data(self, input_data, output_data):
        if not self.save or self.collection_completed:
            return
            
        remaining_capacity = self.max_save_points - self.saved_points
        
        input_flat = input_data.flatten()
        output_flat = output_data.flatten()
        
        if input_flat.numel() <= remaining_capacity:
            points_to_save = input_flat.numel()
            input_sample = input_flat
            output_sample = output_flat
        else:
            points_to_save = remaining_capacity
            indices = torch.linspace(0, input_flat.numel()-1, points_to_save, dtype=torch.long)
            input_sample = input_flat[indices]
            output_sample = output_flat[indices]
        
        # 保存到统一文件
        os.makedirs("./saved/hist_csvs/", exist_ok=True)
        
        # 所有ADC输入保存到同一个文件
        with open("./saved/hist_csvs/test_resnet20+Cifar10_QAT+cal_input.csv", "a") as f:
            np.savetxt(f, input_sample.cpu().numpy(), delimiter=",")
        
        # 所有ADC输出保存到同一个文件  
        with open("./saved/hist_csvs/test_resnet20+Cifar10_QAT+cal_output.csv", "a") as f:
            np.savetxt(f, output_sample.cpu().numpy(), delimiter=",")
        
        self.saved_points += points_to_save
        
        # 只在首次完成时打印
        if self.saved_points >= self.max_save_points and not self.collection_completed:
            adc_name = getattr(self, 'adc_name', 'unknown')
            print(f"ADC {adc_name}: COMPLETED {self.saved_points}/{self.max_save_points} points")
            self.collection_completed = True
   
    def forward(self, x):
        # Calibration data collection
        if self.calibrating and len(self.cal_data) <30:
            self.cal_data.append(x.flatten().cpu().detach())
        

        #torch.clamp(self.step_size, min=1e-8) 
        # ADC quantization
        original_shape = x.shape
        scale_offset = x / self.step_size - self.zero_point  
        y = (torch.round(scale_offset) * self.step_size).detach() + x - x.detach()
        num_levels = 2 ** self.bits  # 2^bits个量化level
        min_val = 0 * self.step_size
        max_val = (num_levels - 1) * self.step_size
        
         
        min_val = (min_val * self.step_size).clone().detach().to(device=y.device, dtype=y.dtype)
        max_val = (max_val * self.step_size).clone().detach().to(device=y.device, dtype=y.dtype) 
        y = y.clamp(min_val, max_val)
        


        if self.save and self.saved_points < self.max_save_points:
            self._sample_and_save_data(x, y)
        loss = 0  

        if self.custom_loss:
            loss = torch.mean(abs(x - y)**2 ) 
            # if random.random() < 0.001:  
            #     print(f" ADCloss: {loss.item():.6f}")
  
        y = gradientFilter.apply(y, min_val, max_val,self.grad_filter,x)
            
        return y, loss

    def reset_data_collection(self):
        """Reset data collection counters (useful for multiple runs)"""
        self.saved_points = 0
        self.sampling_probability = 1.0

    def get_collection_stats(self):
        """Get statistics about data collection"""
        return {
            'saved_points': self.saved_points,
            'max_points': self.max_save_points,
            'completion_ratio': self.saved_points / self.max_save_points,
            'sampling_probability': self.sampling_probability
        }


class gradientFilter(Function):
    @staticmethod
    def forward(ctx, input_tens, min_val, max_val,grad_filter,raw_tens):
        ctx.grad_filter = grad_filter
        ctx.max_val = max_val
        ctx.min_val = min_val
        ctx.save_for_backward(raw_tens,input_tens)
         
        return input_tens
    
    @staticmethod
    def backward(ctx, grad_output):

        

        max_val = ctx.max_val
        min_val = ctx.min_val
        raw_tens,input_tens = ctx.saved_tensors
    #   # ============ 记录原始梯度统计 ============
    #     original_grad_mean = grad_output.mean().item()
    #     original_grad_std = grad_output.std().item()
    #     original_grad_max = grad_output.abs().max().item()
        if ctx.grad_filter:
            
            scale1 = torch.clamp( 0.2 +torch.log( raw_tens+max_val-1 ) /3, min=0, max=1.0)
            #scale1 = torch.clamp( (0.2 + torch.sqrt(raw_tens - 1)) / 3 , min=0, max=1.0)
             
           # scale3 = torch.clamp(torch.abs(torch.sin(raw_tens * torch.pi)) * 0.8+ 0.2, min=0, max=1.0)
            scale3 =(torch.sin(( raw_tens - 0.25*max_val) * torch.pi * 2/max_val) + 1) * 0.4 + 0.2
            grad_out = torch.where(raw_tens >= min_val, scale3 * grad_output ,0.1 * grad_output)
            grad_out = torch.where(raw_tens < max_val, grad_out,  scale1* grad_output  )
             
            # if random.random() < 0.001:
            #     print(f"With filter - grad_out mean: {grad_out.mean():.6f}, std: {grad_out.std():.6f}")
        else:
             
            grad_out = torch.where(raw_tens >  min_val,  grad_output, 0.1 * grad_output)
            grad_out = torch.where(raw_tens <=   max_val , grad_out,0.1 * grad_output)
          # ============ 记录过滤后的梯度统计 ============
        # filtered_grad_mean = grad_out.mean().item()
        # filtered_grad_std = grad_out.std().item()
        # filtered_grad_max = grad_out.abs().max().item()
        
        # # ============ 保存到文件 ============
        # import os
        # import random
        # if random.random() < 0.001:  # 采样1%的batch
        #     os.makedirs("./saved/gradient_analysis/", exist_ok=True)
        #     with open("./saved/gradient_analysis/gradient_stats.csv", "a") as f:
        #         # 写入: filter_enabled, orig_mean, orig_std, orig_max, filt_mean, filt_std, filt_max
        #         f.write(f"{ctx.grad_filter},{original_grad_mean:.6f},{original_grad_std:.6f},"
        #                f"{original_grad_max:.6f},{filtered_grad_mean:.6f},{filtered_grad_std:.6f},"
        #                f"{filtered_grad_max:.6f}\n")
        
        # return   grad_out,None, None, None, None
            
            
            # if random.random() < 0.001:
            #     print(f"Without filter - grad_out mean: {grad_out.mean():.6f}, std: {grad_out.std():.6f}")
 

        return grad_out, None, None, None, None

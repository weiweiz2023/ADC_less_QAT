import numpy as np
import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from torchviz import make_dot
import random
from src.adc_module import Nbit_ADC
import time
from torch.autograd import Function

def bit_slicing(weight, frac_bit, bit_slice=1, weight_bits=4):
    weight = weight.T.clamp(0.0, 1.0)
    max_int = 2 ** weight_bits - 1
    int_repr = torch.round(weight * max_int).int()
    mask = 1 << torch.arange(weight_bits, device=int_repr.device).flip(0)  
    bits = int_repr.unsqueeze(-1).bitwise_and(mask).ne(0).float()
    binary =bits.reshape(int_repr.shape[0], -1)
    return binary



def float_to_16bits_tensor_fast(input, frac_bits, bit_slice, bit_slice_num, input_bits):
   
    int_bits = input_bits - frac_bits - 1
    max_val = 2**int_bits - 2**(-frac_bits)
    
    input_quantized = torch.round(
        torch.clamp(input, -2**int_bits, max_val) * (2**frac_bits)
    ).to(torch.int32)
    
    mask = input_quantized < 0
    input_quantized = torch.where(
        mask,
        (1 << input_bits) + input_quantized,
        input_quantized
    )
    shifts = torch.arange(
        start=(bit_slice_num-1)*bit_slice,
        end=-1,
        step=-bit_slice,
        device=input.device
    )
    slices = (input_quantized.unsqueeze(-1) >> shifts) & ((1 << bit_slice) - 1)
    
    return slices.float()




# def mvm_tensor(zeros, shift_add_bit_stream, shift_add_bit_slice, output_reg, flatten_input, flatten_input_sign,  
#                xbars, bit_slice, bit_stream, weight_bits, weight_bit_frac, input_bits, input_bit_frac, adc_bit, acm_bit, acm_bit_frac,subarray_size,adc_grad_filter): 
    
#     xbars_row = xbars.shape[0]
#     batch_size = flatten_input.shape[0]
#     bit_stream_num = input_bits//bit_stream
#     Nstates_slice = 2 ** bit_slice - 1
#     Nstates_stream = 2 ** bit_stream - 1

#     adc = Nbit_ADC(adc_bit, Nstates_slice, Nstates_stream,adc_grad_filter)
    
#     if bit_stream == 1:
#         for i in range(bit_stream_num): # 16bit input
#                 input_stream = flatten_input[:,:,:,-1-i].reshape((batch_size, xbars_row, 1, subarray_size, 1))
#             #####
#             #with torch.set_grad_enabled(True): 
#                 start_time = time.time()
#                 output_analog = torch.mul(xbars, input_stream)
#                 output_analog = torch.sum(output_analog,3)
#               #  print(f"Time @ args: {time.time() - start_time}")
#             #####
#                 #print(f"Before ADC: output_analog.requires_grad={output_analog.requires_grad}")
#                 output_analog = adc(output_analog)
#                 #print(f"After ADC: output_analog.requires_grad={output_analog.requires_grad}")

#                 output_analog = output_analog.reshape(shift_add_bit_slice.shape)  # for 32-fixed
                
                
#                 output_analog= torch.sum(torch.mul(output_analog, shift_add_bit_slice), 4)
#                 output_reg[:,:,:,i,:] =output_analog 
#                 output = torch.sum(torch.mul(output_reg, shift_add_bit_stream), 3)

                 
#                 scale_factor = 2**(input_bit_frac + weight_bit_frac - acm_bit_frac)
#                 output = output / scale_factor
     
#                 output_truncated = torch.trunc(output)
#                 output = output + (output_truncated - output).detach()  # STE
#                 mod_factor = 2**acm_bit
#                 output_mod = torch.fmod(output, mod_factor)
#                 output = output + (output_mod - output).detach()  # STE
#                 output = output / (2**acm_bit_frac)

#         output = torch.sum(output, 1)
        
#         output = output.reshape(batch_size, -1)
#         output = output.float()
#     # print(f"mvm_tensor return: requires_grad={output.requires_grad}")
#     # print(f"mvm_tensor return: grad_fn={output.grad_fn}")
#     # if output.grad_fn is None and output.requires_grad:
#     #     print("⚠️ WARNING: Tensor requires_grad but has no grad_fn (detached!)")    
#     return output 
    
class STE_Quantize(Function):
    @staticmethod
    def forward(ctx, input_tens, bits):
        ctx.save_for_backward(input_tens)
        if bits > 1:
            neg_2s = 2 ** (bits - 1)
            pos_2s = neg_2s - 1
            if pos_2s == 0:  # 1-bit -> 1.58-bit
                pos_2s = 1
            negative_tensor = torch.where(input_tens < 0, input_tens, 0)
            positive_tensor = torch.where(input_tens > 0, input_tens, 0)
            negative_tensor = torch.round(negative_tensor * neg_2s) / neg_2s
            positive_tensor = torch.round(positive_tensor * pos_2s) / pos_2s
            out = (positive_tensor + negative_tensor)
        elif bits == 1:
            out = torch.where(input_tens < 0, torch.floor(input_tens), input_tens)
            out = torch.where(out > 0, torch.ceil(out), out)
        else:
            raise ValueError("Weight bits cannot be <= 0")

        out = torch.clamp(out, -1, 1)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        input_tens, = ctx.saved_tensors
        
        # Clamp ends (-3, 3) to avoid exploding gradients
        grad_out = torch.where(input_tens > -1, grad_output, 0)
        grad_out = torch.where(input_tens < 1, grad_out, 0)
        
        # Implement straight through estimator (STE)
        # grad_out = grad_output
        return grad_out, None
    
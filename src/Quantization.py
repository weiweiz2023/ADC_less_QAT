import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from src.adc_module import Nbit_ADC
import random



class WeightQuantization(Function):
    @staticmethod
    def forward(ctx, weights, weight_bits):
        ctx.save_for_backward(weights)
        ctx.weight_bits = weight_bits
        
        # Split into positive and negative components
        weights_pos = torch.clamp(weights, min=0.0)
        weights_neg = torch.clamp(-weights, min=0.0)
        
        # Normalize by max absolute value
        max_abs = max(weights_pos.max(), weights_neg.max())
        ctx.max_abs = max_abs
        
        if max_abs > 0:
            weights_pos = weights_pos / max_abs
            weights_neg = weights_neg / max_abs
        else:
            max_abs = 1.0
        
        # Quantize to integer representation (0 to 2^weight_bits - 1)
        max_val_pos =2 ** (weight_bits - 1)  
        max_val_neg = max_val_pos - 1
        pos_quantized = torch.round(weights_pos * max_val_pos)/max_val_pos
        neg_quantized = torch.round(weights_neg * max_val_neg)/max_val_neg
         
        # Reconstruct quantized weights
        qw = (pos_quantized - neg_quantized) * max_abs  
        return qw
    
    @staticmethod
    def backward(ctx, grad_output):
        weights, = ctx.saved_tensors
        max_abs = ctx.max_abs
        
      
        grad_input = torch.where(
            (weights >= -max_abs) & (weights <= max_abs),
            grad_output,
            torch.zeros_like(grad_output)
        )
        if max_abs > 0:
            grad_input = grad_input / max_abs
        return grad_input, None


class InputQuantization(Function):
    @staticmethod
    def forward(ctx, inputs, input_bits, frac_bits):
        ctx.save_for_backward(inputs)
        ctx.input_bits = input_bits
        ctx.frac_bits = frac_bits
        
        # Calculate integer bits and value range
        int_bits = input_bits - frac_bits - 1  # -1 for sign bit
        max_val = 2**int_bits - 2**(-frac_bits)
        min_val = -2**int_bits
        ctx.max_val = max_val
        ctx.min_val = min_val
        
        # Quantize: clamp, scale by fractional precision, and round
        inputs_quantized = torch.round(
            torch.clamp(inputs, min_val, max_val) * (2**frac_bits)
        )
       
        # Scale back to float range
        inputs_quantized = inputs_quantized / (2**frac_bits)
        return inputs_quantized
    
    @staticmethod
    def backward(ctx, grad_output):
        inputs, = ctx.saved_tensors
        max_val = ctx.max_val
        min_val = ctx.min_val
        
        # 修正：使用实际的量化范围
        grad_input = torch.where(
            (inputs >= min_val) & (inputs <= max_val),
            grad_output,
            torch.zeros_like(grad_output)
        )
        
        return grad_input, None, None



 #============================bitslice stream=================================  
class VectorizedWeightBitSlicing(Function):
    @staticmethod
    def forward(ctx, weights, weight_bits, bits_per_slice):
        ctx.save_for_backward(weights)
        ctx.weight_bits = weight_bits
        ctx.bits_per_slice = bits_per_slice
        weights_pos = torch.clamp(weights, min=0.0)
        weights_neg = torch.clamp(-weights, min=0.0)
        
        max_abs = max(weights_pos.max(), weights_neg.max())
        ctx.max_abs = max_abs
        if max_abs > 0:
            weights_pos = weights_pos / max_abs
            weights_neg = weights_neg / max_abs
        else:
            max_abs = 1.0


        max_val_pos =2 ** (weight_bits - 1)  
        max_val_neg = max_val_pos - 1
        pos_int = torch.round(weights_pos * max_val_pos).int()
        neg_int = torch.round(weights_neg * max_val_neg).int()
        
        num_slices = weight_bits // bits_per_slice
        bit_positions = torch.arange(0, weight_bits, bits_per_slice, device=weights.device)
        
        pos_slices = (pos_int.unsqueeze(-1) >> bit_positions) & ((1 << bits_per_slice) - 1)
        neg_slices = (neg_int.unsqueeze(-1) >> bit_positions) & ((1 << bits_per_slice) - 1)
        
        return pos_slices.float(), neg_slices.float(), max_abs
    
    @staticmethod
    def backward(ctx, grad_pos, grad_neg, grad_norm):
        weights, = ctx.saved_tensors
        max_abs = ctx.max_abs
        grad_input = torch.where(weights >= 0, 
                                grad_pos.sum(dim=-1), 
                                -grad_neg.sum(dim=-1))
      #  grad_input = torch.clamp(grad_input, -1.0, 1.0)
        
        return grad_input, None, None
# class VectorizedInputBitStreaming(Function):
#     @staticmethod
#     def forward(ctx, inputs, input_bits, frac_bits, bits_per_stream, num_streams):
#         ctx.save_for_backward(inputs)
#         ctx.input_bits = input_bits
#         ctx.frac_bits = frac_bits
#         ctx.bits_per_stream = bits_per_stream
#         int_bits = input_bits - frac_bits 
#         max_val = 2**int_bits - 2**(-frac_bits)
#         ctx.max_val = max_val
#         inputs_quantized = torch.round(
#             torch.clamp(inputs, 0, max_val) * (2**frac_bits-1)).to(torch.int32)
        
#         bit_positions = torch.arange(0, input_bits, bits_per_stream, device=inputs.device)
#         streams = ( inputs_quantized.unsqueeze(-1) >> bit_positions) & ((1 << bits_per_stream) - 1)

#         return streams.float()
    
#     @staticmethod
#     def backward(ctx, grad_output):
#         inputs, = ctx.saved_tensors
#         int_bits = ctx.input_bits - ctx.frac_bits  
#         min_val = 0
#         max_val = 2**int_bits - 2**(-ctx.frac_bits)
#         grad_input = torch.where(inputs >min_val, grad_output.sum(dim=-1), 0)
#         grad_input = torch.where(inputs < max_val, grad_input, 0)
#      #   grad_input = torch.clamp(grad_input, -1.0, 1.0)
#         return grad_input, None, None, None, None
class VectorizedInputBitStreaming(Function):
    @staticmethod
    def forward(ctx, inputs, input_bits, frac_bits, bits_per_stream, num_streams):
        ctx.save_for_backward(inputs)
        ctx.input_bits = input_bits
        ctx.frac_bits = frac_bits
        ctx.bits_per_stream = bits_per_stream
        int_bits = input_bits - frac_bits - 1
        max_val = 2**int_bits - 2**(-frac_bits)
        ctx.max_val = max_val
        inputs_quantized = torch.round(
            torch.clamp(inputs, -2**int_bits, max_val) * (2**frac_bits)).to(torch.int32)
        
        negative_mask = inputs_quantized < 0
        inputs_quantized = torch.where(negative_mask,
                                     (1 << input_bits) + inputs_quantized,
                                     inputs_quantized)
        
        bit_positions = torch.arange(0, input_bits, bits_per_stream, device=inputs.device)
        streams = (inputs_quantized.unsqueeze(-1) >> bit_positions) & ((1 << bits_per_stream) - 1)
        
        return streams.float()
    
    @staticmethod
    def backward(ctx, grad_output):
        inputs, = ctx.saved_tensors
        int_bits = ctx.input_bits - ctx.frac_bits - 1
        min_val = -2**int_bits
        max_val = 2**int_bits - 2**(-ctx.frac_bits)
        grad_input = torch.where(inputs >min_val, grad_output.sum(dim=-1), 0)
        grad_input = torch.where(inputs < max_val, grad_input, 0)
        #grad_input = torch.clamp(grad_input, -1.0, 1.0)
        return grad_input, None, None, None, None


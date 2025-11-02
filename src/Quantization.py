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



 #============ ================bitslice stream=================================  
# class VectorizedWeightBitSlicing(Function):
#     @staticmethod
#     def forward(ctx, weights, weight_bits, bits_per_slice):
#         """
#         正负分离的双极性权重bit slicing
#         分别对正负权重进行bit slicing，然后相减
#         """
#         ctx.save_for_backward(weights)
#         ctx.weight_bits = weight_bits
#         ctx.bits_per_slice = bits_per_slice
        
#         #  Per-channel归一化
#         max_abs = weights.abs().amax(dim=(1, 2, 3), keepdim=True)
#         max_abs = torch.clamp(max_abs, min=1e-8)
#         ctx.max_abs = max_abs
        
#         # 归一化到 [-1, 1]
#         weights_normalized = weights / max_abs
        
#         #  量化到有符号整数 [0, 2^(n-1)-1]（注意：这里用无符号范围）
#         max_int = 2 ** (weight_bits - 1) - 1
#         weights_scaled = weights_normalized * max_int
        
#         # 分离正负权重
#         # 正值部分：负值变0
#         weights_positive = torch.clamp(weights_scaled, min=0).round().int()
        
#         # 负值部分：正值变0，负值取绝对值
#         weights_negative = torch.clamp(-weights_scaled, min=0).round().int()
        
#         #   Bit slicing - 正值
#         bit_positions = torch.arange(0, weight_bits, bits_per_slice, device=weights.device)
#         num_slices = len(bit_positions)
        
#         # 正值bit slicing
#         positive_slices = (weights_positive.unsqueeze(-1) >> bit_positions) & ((1 << bits_per_slice) - 1)
        
#         # 负值bit slicing
#         negative_slices = (weights_negative.unsqueeze(-1) >> bit_positions) & ((1 << bits_per_slice) - 1)
        
#         #   相减得到有符号的bit slices  
#         weight_slices = positive_slices.float() - negative_slices.float()
        
#         # 保存正负slice用于梯度计算
#         ctx.positive_slices = positive_slices
#         ctx.negative_slices = negative_slices
        
#         return weight_slices, max_abs
    
#     @staticmethod
#     def backward(ctx, grad_slices, grad_norm):
#         """
#         正负分离的梯度反向传播
#         需要补偿forward中的scale放大
#         """
#         weights, = ctx.saved_tensors
#         max_abs = ctx.max_abs
        
#         #   grad_slices shape: [out_ch, in_ch, K, K, num_slices]
#         # 每个slice在forward中被乘以了 2^(slice_idx * bits_per_slice)
        
#         # 计算每个slice的scale (用于反向补偿)
#         bits_per_slice = ctx.bits_per_slice
#         num_slices = grad_slices.shape[-1]
        
#         # slice_weights = [1, 2^bits_per_slice, 2^(2*bits_per_slice), ...]
#         slice_weights = 2.0 ** (torch.arange(num_slices, device=weights.device) * bits_per_slice)
#         slice_weights = slice_weights.view(1, 1, 1, 1, -1)  # broadcast shape
        
#         # ⭐ 除以scale来补偿forward的放大
#         grad_slices_normalized = grad_slices / slice_weights
        
#         # 对所有slice求和
#         grad_combined = grad_slices_normalized.sum(dim=-1)
    
#         weight_max_int = 2 ** (ctx.weight_bits - 1) - 1
#         grad_combined = grad_combined / weight_max_int
        
#         # STE
#         grad_input = torch.where(
#             torch.abs(weights) <= max_abs,
#             grad_combined,
#             0.1 * grad_combined
#         )
        
#         return grad_input, None, None



# class VectorizedInputBitStreaming(Function):
#     @staticmethod
#     def forward(ctx, inputs, input_bits, frac_bits, bits_per_stream, num_streams):
#         ctx.save_for_backward(inputs)
#         ctx.input_bits = input_bits
#         ctx.frac_bits = frac_bits
#         ctx.bits_per_stream = bits_per_stream
#         int_bits = input_bits - frac_bits - 1
#         max_val = 2**int_bits - 2**(-frac_bits)
#         ctx.max_val = max_val
#         inputs_quantized = torch.round(
#             torch.clamp(inputs, -2**int_bits, max_val) * (2**frac_bits)).to(torch.int32)
        
#         negative_mask = inputs_quantized < 0
#         inputs_quantized = torch.where(negative_mask,
#                                      (1 << input_bits) + inputs_quantized,
#                                      inputs_quantized)
        
#         bit_positions = torch.arange(0, input_bits, bits_per_stream, device=inputs.device)
#         streams = (inputs_quantized.unsqueeze(-1) >> bit_positions) & ((1 << bits_per_stream) - 1)
        
#         return streams.float()
    
#     @staticmethod
#     def backward(ctx, grad_output):
#         """
#         Input bit streaming的梯度
#         需要补偿stream scale
#         """
#         inputs, = ctx.saved_tensors
        
#         # grad_output shape: [..., num_streams]
#         bits_per_stream = ctx.bits_per_stream
#         num_streams = grad_output.shape[-1]
        
#         # 计算每个stream的scale
#         stream_weights = 2.0 ** (torch.arange(num_streams, device=inputs.device) * bits_per_stream)
#         stream_weights = stream_weights.view(*([1] * (grad_output.ndim - 1)), -1)
        
#         # ⭐ 除以scale
#         grad_normalized = grad_output / stream_weights
        
#         # 对所有stream求和
#         grad_combined = grad_normalized.sum(dim=-1)
        
#         # 除以量化scale
#         input_scale = 2 ** ctx.frac_bits
#         grad_combined = grad_combined / input_scale
        
#         # STE
#         int_bits = ctx.input_bits - ctx.frac_bits - 1
#         min_val = -2**int_bits
#         max_val = 2**int_bits - 2**(-ctx.frac_bits)
        
#         grad_input = torch.where(
#             (inputs >= min_val) & (inputs <= max_val),
#             grad_combined,
#             0.1 * grad_combined
#         )
        
#         return grad_input, None, None, None, None


class VectorizedWeightBitSlicing(Function):
    @staticmethod
    def forward(ctx, weights, weight_bits, bits_per_slice):
        ctx.save_for_backward(weights)
        ctx.weight_bits = weight_bits
        ctx.bits_per_slice = bits_per_slice
        max_val_pos =2 ** (weight_bits - 1)  
        max_val_neg = max_val_pos - 1
        weights_pos = torch.clamp(weights, min=0.0)
        weights_neg = torch.clamp(-weights, min=0.0)
        #alpha = 2 * weights.abs().mean() / torch.sqrt( max_val_pos)
        max_pos =  weights_pos.max()  
        max_neg =  weights_neg.max() 
        ctx.max_abs = max_pos
        if max_pos > 0:
            weights_pos = weights_pos / max_pos
            weights_neg = weights_neg / max_neg
        else:
            max_pos = 1.0


        
        pos_int = torch.round(weights_pos * max_val_pos).int()
        neg_int = torch.round(weights_neg * max_val_neg).int()
        
        num_slices = weight_bits // bits_per_slice
        bit_positions = torch.arange(0, weight_bits, bits_per_slice, device=weights.device)
        
        pos_slices = (pos_int.unsqueeze(-1) >> bit_positions) & ((1 << bits_per_slice) - 1)
        neg_slices = (neg_int.unsqueeze(-1) >> bit_positions) & ((1 << bits_per_slice) - 1)
        
        return pos_slices.float(), neg_slices.float(), max_pos
    
    @staticmethod
    def backward(ctx, grad_pos, grad_neg, grad_norm):
        weights, = ctx.saved_tensors
        max_abs = ctx.max_abs
        
        grad_input = torch.where(weights >= 0, 
                                grad_pos.sum(dim=-1), 
                                -grad_neg.sum(dim=-1))
      #  grad_input = torch.clamp(grad_input, -1.0, 1.0)
        
        return grad_input, None, None



# class VectorizedWeightBitSlicing(Function):
#     @staticmethod
#     def forward(ctx, weights, weight_bits, bits_per_slice):
#         ctx.save_for_backward(weights)
#         ctx.weight_bits = weight_bits
#         ctx.bits_per_slice = bits_per_slice
#         max_val_pos =2 ** (weight_bits - 1)  
#         max_val_neg = max_val_pos - 1
#         weights_pos = torch.clamp(weights, min=0.0)
#         weights_neg = torch.clamp(-weights, min=0.0)
#         #alpha = 2 * weights.abs().mean() / torch.sqrt( max_val_pos)

#         max_abs = max(weights_pos.max(), weights_neg.max())
#         ctx.max_abs = max_abs
#         if max_abs > 0:
#             weights_pos = weights_pos / max_abs
#             weights_neg = weights_neg / max_abs
#         else:
#             max_abs = 1.0


        
#         pos_int = torch.round(weights_pos * max_val_pos).int()
#         neg_int = torch.round(weights_neg * max_val_neg).int()
        
#         num_slices = weight_bits // bits_per_slice
#         bit_positions = torch.arange(0, weight_bits, bits_per_slice, device=weights.device)
        
#         pos_slices = (pos_int.unsqueeze(-1) >> bit_positions) & ((1 << bits_per_slice) - 1)
#         neg_slices = (neg_int.unsqueeze(-1) >> bit_positions) & ((1 << bits_per_slice) - 1)
        
#         return pos_slices.float(), neg_slices.float(), max_abs
    
#     @staticmethod
#     def backward(ctx, grad_pos, grad_neg, grad_norm):
#         weights, = ctx.saved_tensors
#         max_abs = ctx.max_abs
        
#         grad_input = torch.where(weights >= 0, 
#                                 grad_pos.sum(dim=-1), 
#                                 -grad_neg.sum(dim=-1))
#       #  grad_input = torch.clamp(grad_input, -1.0, 1.0)
        
#         return grad_input, None, None




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
        grad_input = torch.clamp(grad_input, -1.0, 1.0)
        return grad_input, None, None, None, None



import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from src.adc_module import Nbit_ADC

class VectorizedWeightBitSlicing(Function):
    @staticmethod
    def forward(ctx, weights, weight_bits, bits_per_slice):
        ctx.save_for_backward(weights)
        ctx.weight_bits = weight_bits
        
        weights_pos = torch.clamp(weights, min=0.0)
        weights_neg = torch.clamp(-weights, min=0.0)
        
      
        max_abs =max(weights_pos.max(), weights_neg.max())
        if max_abs > 0:
            weights_pos = weights_pos / max_abs
            weights_neg = weights_neg / max_abs
        else:
            max_abs = 1.0
        
        max_val = (2 ** weight_bits) - 1
        pos_int = torch.round(weights_pos * max_val).int()
        neg_int = torch.round(weights_neg * max_val).int()
        
        num_slices = weight_bits // bits_per_slice
        bit_positions = torch.arange(0, weight_bits, bits_per_slice, device=weights.device)
        
        pos_slices = (pos_int.unsqueeze(-1) >> bit_positions) & ((1 << bits_per_slice) - 1)
        neg_slices = (neg_int.unsqueeze(-1) >> bit_positions) & ((1 << bits_per_slice) - 1)
        
        return pos_slices.float(), neg_slices.float(), max_abs
    
    @staticmethod
    def backward(ctx, grad_pos, grad_neg, grad_norm):
        weights, = ctx.saved_tensors
        grad_input = torch.where(weights >= 0, 
                                grad_pos.mean(dim=-1), 
                                -grad_neg.mean(dim=-1))
        grad_input = torch.clamp(grad_input, -1.0, 1.0)
        return grad_input, None, None

class VectorizedInputBitStreaming(Function):
   
    @staticmethod
    def forward(ctx, inputs, input_bits, frac_bits, bits_per_stream, num_streams):
        ctx.save_for_backward(inputs)
        ctx.input_bits = input_bits
        ctx.frac_bits = frac_bits
        
        int_bits = input_bits - frac_bits - 1
        max_val = 2**int_bits - 2**(-frac_bits)
        
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
        grad_input = torch.where(inputs > min_val, grad_output.mean(dim=-1), 0)
        grad_input = torch.where(inputs < max_val, grad_input, 0)
       
        grad_input = torch.clamp(grad_input, -1.0, 1.0)
        return grad_input, None, None, None, None

class quantized_conv(nn.Module):
    def __init__(self, in_channels, out_channels, arch_args, 
                 kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=None):
        super(quantized_conv, self).__init__()
        
        self.weight_bits = arch_args.weight_bits
        self.weight_bits_per_slice = arch_args.bit_slice
        self.weight_slices = max(1, self.weight_bits // max(self.weight_bits_per_slice, 1))
        self.weight_frac_bits = arch_args.weight_bit_frac
        
        self.input_bits = arch_args.input_bits
        self.input_bits_per_stream = arch_args.bit_stream
        self.input_streams = max(1, self.input_bits // max(self.input_bits_per_stream, 1))
        self.input_frac_bits = arch_args.input_bit_frac
        
        # ADC
        self.adc_bits = arch_args.adc_bit
        self.adc_grad_filter = arch_args.adc_grad_filter
        self.save_adc_data = arch_args.save_adc
        self.adc_custom_loss = arch_args. adc_custom_loss
        self.adc_reg_lambda = arch_args.adc_reg_lambda
        # conv
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = (stride, stride)
        self.padding = (padding, padding)
        self.dilation = (dilation, dilation)
        self.groups = groups
        
        subarray_size = arch_args.subarray_size
        if subarray_size <= 0:
            self.num_subarrays = 0
        else:
            total_inputs = in_channels * (kernel_size ** 2)
            self.num_subarrays = max(1, (total_inputs + subarray_size - 1) // subarray_size)
        
        self.experiment_state = arch_args.experiment_state
        
        # 权重参数
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, kernel_size, kernel_size))
        nn.init.kaiming_normal_(self.weight)
        
        
        
        self.adc_pos = Nbit_ADC(self.adc_bits, self.weight_slices, self.input_streams, 
                               self.save_adc_data, self.adc_grad_filter,
                                self.adc_custom_loss, self.adc_reg_lambda)
        self.adc_neg = Nbit_ADC(self.adc_bits, self.weight_slices, self.input_streams, 
                               self.save_adc_data, self.adc_grad_filter,
                                self.adc_custom_loss, self.adc_reg_lambda)
    
    def compute_vectorized_conv(self, inputs, weights):
        import time
        t1 = time.time()
        input_patches = F.unfold(inputs, self.kernel_size, self.dilation, self.padding, self.stride)
        batch_size, patch_features, num_patches = input_patches.shape
        
        t1 = time.time()
        input_streams = VectorizedInputBitStreaming.apply(
                input_patches, self.input_bits, self.input_frac_bits,
                self.input_bits_per_stream, self.input_streams
            )
        
        t1 = time.time()
        pos_slices, neg_slices, norm_factor = VectorizedWeightBitSlicing.apply(
            weights, self.weight_bits, self.weight_bits_per_slice
            )
        
        pos_weight_matrix = pos_slices.view(self.out_channels, -1, self.weight_slices)
        neg_weight_matrix = neg_slices.view(self.out_channels, -1, self.weight_slices)
        
        t1 = time.time()
        total_adc_loss = torch.tensor(0.0, device=inputs.device)  # 初始化总ADC损失
        
        if self.num_subarrays > 1:
            input_chunks = torch.chunk(input_streams, self.num_subarrays, dim=1)
            pos_chunks = torch.chunk(pos_weight_matrix, self.num_subarrays, dim=1)
            neg_chunks = torch.chunk(neg_weight_matrix, self.num_subarrays, dim=1)
            
            results = []
            for input_chunk, pos_chunk, neg_chunk in zip(input_chunks, pos_chunks, neg_chunks):
                chunk_result, chunk_loss = self._process_subarray_vectorized(
                    input_chunk, pos_chunk, neg_chunk, batch_size
                )
                results.append(chunk_result)
                total_adc_loss = total_adc_loss + chunk_loss  # 累积损失
            
            final_output = torch.stack(results, dim=0).sum(dim=0)
        else:
            final_output, total_adc_loss = self._process_subarray_vectorized(
                input_streams, pos_weight_matrix, neg_weight_matrix, batch_size
            )
        
        final_output = final_output * norm_factor
        input_scale = 2 ** self.input_bits - 1
        final_output = final_output / input_scale
        
        output_h = self._calc_output_size(inputs.shape[2], 0)
        output_w = self._calc_output_size(inputs.shape[3], 1)
        output = F.fold(final_output, (output_h, output_w), (1, 1))
        
        return output, total_adc_loss  # 返回输出和总ADC损失
    # def _process_subarray_vectorized(self, input_chunk, weight_pos_chunk, weight_neg_chunk, batch_size):
    #     """紧急版本 - 最小内存使用"""
    #     device = input_chunk.device
        
    #     # 直接使用原始Einstein summation但立即处理
    #     pos_results = torch.einsum('bfps,oft->bopst', input_chunk, weight_pos_chunk)
    #     neg_results = torch.einsum('bfps,oft->bopst', input_chunk, weight_neg_chunk)
        
    #     # 立即处理，不保存中间结果
    #     results = pos_results - neg_results
    #     del pos_results, neg_results
        
    #     # 一次性ADC处理
    #     original_shape = results.shape
    #     results_flat = results.flatten()
    #     del results
        
    #     quantized, adc_loss = self.adc_pos(results_flat)
    #     del results_flat
        
    #     quantized = quantized.view(original_shape)
        
    #     # 应用bit权重
    #     stream_weights = 2.0 ** (torch.arange(self.input_streams, device=device) * self.input_bits_per_stream)
    #     slice_weights = 2.0 ** (torch.arange(self.weight_slices, device=device) * self.weight_bits_per_slice)
        
    #     stream_weights = stream_weights.view(1, 1, 1, -1, 1)
    #     slice_weights = slice_weights.view(1, 1, 1, 1, -1)
        
    #     scaled = quantized * stream_weights * slice_weights
    #     del quantized
        
    #     final_output = scaled.sum(dim=(-2, -1))
    #     del scaled
        
    #     return final_output, adc_loss
 
    
    def _process_subarray_vectorized(self, input_chunk, weight_pos_chunk, weight_neg_chunk, batch_size):
        
        
        
        device = input_chunk.device
        stream_weights = 2.0 ** (torch.arange(self.input_streams, device=device) * self.input_bits_per_stream)
        slice_weights = 2.0 ** (torch.arange(self.weight_slices, device=device) * self.weight_bits_per_slice)
        
        # Einstein summation
        pos_results = torch.einsum('bfps,oft->bopst', input_chunk, weight_pos_chunk)
        neg_results = torch.einsum('bfps,oft->bopst', input_chunk, weight_neg_chunk)
        
        if torch.any(neg_results.abs() > 100):
            print(f"WARNING: Large intermediate values detected: max={neg_results.abs().max()}")
            pos_results = torch.clamp(pos_results, -50, 50)
            neg_results = torch.clamp(neg_results, -50, 50)

        original_shape = pos_results.shape
        pos_flat = pos_results.flatten()
        neg_flat = neg_results.flatten()
        
        # 调用ADC并收集损失
        pos_quantized, pos_loss = self.adc_pos(pos_flat)
        neg_quantized, neg_loss = self.adc_neg(neg_flat)
        
        pos_quantized = pos_quantized.view(original_shape)
        neg_quantized = neg_quantized.view(original_shape)
        
        # 计算总ADC损失
        total_adc_loss = pos_loss + neg_loss
        
        stream_weights = stream_weights.view(1, 1, 1, -1, 1)  # [1,1,1,streams,1]
        slice_weights = slice_weights.view(1, 1, 1, 1, -1)   # [1,1,1,1,slices]
        torch.cuda.empty_cache()
        
        pos_scaled = pos_quantized * stream_weights * slice_weights
        neg_scaled = neg_quantized * stream_weights * slice_weights
        
        pos_final = pos_scaled.sum(dim=(-2, -1))  # [batch, out_ch, patches]
        neg_final = neg_scaled.sum(dim=(-2, -1))
        
        return pos_final - neg_final, total_adc_loss  # 返回结果和ADC损失
    
    def _calc_output_size(self, input_size, dim):
        kernel = self.kernel_size
        pad = self.padding[dim]
        dilation = self.dilation[dim]
        stride = self.stride[dim]
        return (input_size + 2 * pad - dilation * (kernel - 1) - 1) // stride + 1
    
    def forward(self, inputs):
        if self.experiment_state == "PTQAT" and self.num_subarrays > 0:
            if self.weight_bits > 0 or self.input_bits > 0:
                output, adc_loss = self.compute_vectorized_conv(inputs, self.weight)
                return output, adc_loss  # 返回输出和ADC损失
            else:
                output = F.conv2d(inputs, self.weight, bias=None,
                              stride=self.stride, padding=self.padding,
                              dilation=self.dilation, groups=self.groups)
                return output, torch.tensor(0.0, device=inputs.device)  # 无量化时返回0损失
        else:
            output = F.conv2d(inputs, self.weight, bias=None,
                          stride=self.stride, padding=self.padding,
                          dilation=self.dilation, groups=self.groups)
            return output, torch.tensor(0.0, device=inputs.device)  # 无量化时返回0损失


def verify_vectorization(original_conv, vectorized_conv, test_input):
    original_conv.eval()
    vectorized_conv.eval()
    
    with torch.no_grad():
        vectorized_conv.weight.data.copy_(original_conv.weight.data)
        
        out1 = original_conv(test_input)
        out2 = vectorized_conv(test_input)
        
        max_diff = torch.max(torch.abs(out1 - out2))
        rel_diff = max_diff / torch.max(torch.abs(out1))
        
        print(f"最大绝对差异: {max_diff:.8f}")
        print(f"最大相对差异: {rel_diff:.8f}")
        
        if max_diff < 1e-5 or rel_diff < 1e-4:
            print("✓ 向量化验证通过")
            return True
        else:
            print("✗ 向量化验证失败")
            return False
        























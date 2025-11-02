import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from src.adc_module import Nbit_ADC
import random
from src.Quantization import InputQuantization, WeightQuantization, VectorizedWeightBitSlicing, VectorizedInputBitStreaming
import math


# class quantized_conv(nn.Module):
#     def __init__(self, in_channels, out_channels, arch_args, 
#                  kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=None):
#         super(quantized_conv, self).__init__()
        
#         self.weight_bits = arch_args.weight_bits
#         self.weight_bits_per_slice = arch_args.bit_slice
#         self.weight_slices = max(1, self.weight_bits // max(self.weight_bits_per_slice, 1))
#         self.weight_frac_bits = arch_args.weight_bit_frac
        
#         self.input_bits = arch_args.input_bits
#         self.input_bits_per_stream = arch_args.bit_stream
#         self.input_streams = max(1, self.input_bits // max(self.input_bits_per_stream, 1))
#         self.input_frac_bits = arch_args.input_bit_frac
        
#         # ADC
#         self.adc_bits = arch_args.adc_bit
#         self.adc_grad_filter = arch_args.adc_grad_filter
#         self.save_adc_data = arch_args.save_adc
#         self.adc_custom_loss = arch_args.adc_custom_loss
#         self.adc_reg_lambda = arch_args.adc_reg_lambda
        
#         # conv
#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         self.kernel_size = kernel_size
#         self.stride = (stride, stride)
#         self.padding = (padding, padding)
#         self.dilation = (dilation, dilation)
#         self.groups = groups

#         self.layer_name = None  # 会在外部设置
#         self.stats_list = []

#         subarray_size = arch_args.subarray_size
#         if subarray_size <= 0:
#             self.num_subarrays = 0
#         else:
#             total_inputs = in_channels * (kernel_size ** 2)
#             self.num_subarrays = max(1, (total_inputs + subarray_size - 1) // subarray_size)
        
#         self.experiment_state = arch_args.experiment_state
        
#         # Weight parameters
#         self.weight = nn.Parameter(torch.empty(out_channels, in_channels, kernel_size, kernel_size))
#         nn.init.kaiming_normal_(self.weight)
        
       
#         # ⭐ 只需要一个ADC（不分正负）
#         self.adc = Nbit_ADC(
#             self.adc_bits, self.weight_slices, self.input_streams, 
#             self.save_adc_data, self.adc_grad_filter,
#             self.adc_custom_loss, self.adc_reg_lambda
#         )
        
#         # Precompute scale factors
#         stream_weights = 2.0 ** (torch.arange(self.input_streams) * self.input_bits_per_stream)
#         slice_weights = 2.0 ** (torch.arange(self.weight_slices) * self.weight_bits_per_slice)
        
#         # ⭐ 符号位的权重是负的
       
        
#         self.register_buffer('stream_scale', stream_weights.abs().view(1, 1, 1, -1, 1))
#         self.register_buffer('slice_scale', slice_weights.abs().view(1, 1, 1, 1, -1))
    

#     def _fix_input_sign_bit(self, ps):
#         """
#         修正补码input的符号位
        
#         补码的符号位(MSB)权重应该是负的，但MVM按正数算了
#         解决：把包含符号位的stream取反
        
#         Args:
#             ps: [batch, out_ch, patches, streams, slices]
#         Returns:
#             修正后的ps（现在input是真正的有符号数）
#         """
#         if self.input_bits_per_stream != 1:
#             print(f"⚠️ Warning: input bits_per_stream={self.input_bits_per_stream} != 1, "
#                   f"sign handling may be inaccurate")
#             return ps
        
#         sign_stream_idx = self.input_bits - 1
#         result = ps.clone()
#         result[:, :, :, sign_stream_idx, :] = -ps[:, :, :, sign_stream_idx, :]
#         return result
    
#     def compute_vectorized_conv(self, inputs, weights):
#         """
#         统一的卷积计算（不分正负）
#         """
#         # Unfold
#         input_patches = F.unfold(inputs, self.kernel_size, self.dilation, self.padding, self.stride)
        
#         # Input bit streaming
#         input_streams = VectorizedInputBitStreaming.apply(
#             input_patches, self.input_bits, self.input_frac_bits,
#             self.input_bits_per_stream, self.input_streams
#         )
       
#         # Weight bit slicing（统一）
#         weight_slices, norm_factor = VectorizedWeightBitSlicing.apply(
#             weights, self.weight_bits, self.weight_bits_per_slice
#         )
        
#         # Reshape
#         weight_matrix = weight_slices.view(self.out_channels, -1, self.weight_slices)
        
#         total_adc_loss = torch.tensor(0.0, device=inputs.device)
        
#         if self.num_subarrays > 1:
#             input_chunks = torch.chunk(input_streams, self.num_subarrays, dim=1)
#             weight_chunks = torch.chunk(weight_matrix, self.num_subarrays, dim=1)
            
#             chunk_results = []
#             for input_chunk, weight_chunk in zip(input_chunks, weight_chunks):
#                 # 单次MVM
#                 results = torch.einsum('bfps,oft->bopst', input_chunk, weight_chunk)
#                 # ⭐ 添加调试
#                 if torch.isnan(results).any() or torch.isinf(results).any():
#                     print(f"❌ NaN/Inf detected in MVM results!")
#                     print(f"   results range: [{results.min():.4f}, {results.max():.4f}]")
#                     print(f"   input_streams range: [{input_streams.min():.4f}, {input_streams.max():.4f}]")
#                     print(f"   weight_matrix range: [{weight_matrix.min():.4f}, {weight_matrix.max():.4f}]")
#                 # ADC量化
#                 quantized, loss = self.adc(results)
#                 total_adc_loss = total_adc_loss + loss
                
#                 # 缩放（符号位已经在slice_scale中处理）
#                 scaled = quantized * self.stream_scale * self.slice_scale
                
                
#                 chunk_result = scaled.sum(dim=(-2, -1))
#                 chunk_results.append(chunk_result)
            
#             final_output = torch.stack(chunk_results, dim=0).sum(dim=0)
#         else:
#             # 单阵列
#             results = torch.einsum('bfps,oft->bopst', input_streams, weight_matrix)
#             # ⭐ 添加调试
#             if torch.isnan(results).any() or torch.isinf(results).any():
#                 print(f"❌ NaN/Inf detected in MVM results!")
#                 print(f"   results range: [{results.min():.4f}, {results.max():.4f}]")
#                 print(f"   input_streams range: [{input_streams.min():.4f}, {input_streams.max():.4f}]")
#                 print(f"   weight_matrix range: [{weight_matrix.min():.4f}, {weight_matrix.max():.4f}]")
#             quantized, loss = self.adc(results)
#             total_adc_loss = loss
            
#             scaled = quantized * self.stream_scale * self.slice_scale
#             final_output = scaled.sum(dim=(-2, -1))
        
#         # Per-channel缩放
#         final_output = final_output * norm_factor.view(1, -1, 1)
        
#         # 量化级数缩放
#         weight_quant_scale = 2 ** (self.weight_bits - 1) - 1  # 正确
#         input_quant_scale = 2 ** (self.input_frac_bits)  #
#         final_output = final_output / (weight_quant_scale * input_quant_scale)
        
#         # Fold
#         output_h = self._calc_output_size(inputs.shape[2], 0)
#         output_w = self._calc_output_size(inputs.shape[3], 1)
#         output = F.fold(final_output, (output_h, output_w), (1, 1))
        
#         return output, total_adc_loss
#     def _calc_output_size(self, input_size, dim):
#         kernel = self.kernel_size
#         pad = self.padding[dim]
#         dilation = self.dilation[dim]
#         stride = self.stride[dim]
#         return (input_size + 2 * pad - dilation * (kernel - 1) - 1) // stride + 1
    
#     def forward(self, inputs):
#         if self.experiment_state == "PTQAT" and self.num_subarrays > 0:
#            ## if self.weight_bits > 0 or self.input_bits > 0:
#                 output, adc_loss = self.compute_vectorized_conv(inputs, self.weight)
#                 return output, adc_loss
#         elif self.experiment_state == "QAT":

#                 qa = InputQuantization.apply(inputs, self.input_bits, self.input_frac_bits)
#                 qw = WeightQuantization.apply(self.weight, self.weight_bits)
#                 output = F.conv2d(qa, qw , bias=None,
#                               stride=self.stride, padding=self.padding,
#                               dilation=self.dilation, groups=self.groups)
              
#                 return output, torch.tensor(0.0, device=inputs.device)
#         elif self.experiment_state == "pretraining" or self.experiment_state == "pruning":
#             output = F.conv2d(inputs, self.weight, bias=None,
#                           stride=self.stride, padding=self.padding,
#                           dilation=self.dilation, groups=self.groups)
#             return output, torch.tensor(0.0, device=inputs.device)









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
        self.adc_custom_loss = arch_args.adc_custom_loss
        self.adc_reg_lambda = arch_args.adc_reg_lambda
        
        # conv
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = (stride, stride)
        self.padding = (padding, padding)
        self.dilation = (dilation, dilation)
        self.groups = groups

        self.layer_name = None  # 会在外部设置
        self.stats_list = []

        subarray_size = arch_args.subarray_size
        if subarray_size <= 0:
            self.num_subarrays = 0
        else:
            total_inputs = in_channels * (kernel_size ** 2)
            self.num_subarrays = max(1, (total_inputs + subarray_size - 1) // subarray_size)
        
        self.experiment_state = arch_args.experiment_state
        
        # Weight parameters
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, kernel_size, kernel_size))
        nn.init.kaiming_normal_(self.weight)
        
       
       
        self.adc_pos = Nbit_ADC(self.adc_bits, self.weight_slices, self.input_streams, 
                               self.save_adc_data, self.adc_grad_filter,
                               self.adc_custom_loss, self.adc_reg_lambda)
        self.adc_neg = Nbit_ADC(self.adc_bits, self.weight_slices, self.input_streams, 
                               self.save_adc_data, self.adc_grad_filter,
                               self.adc_custom_loss, self.adc_reg_lambda)
        
        # Precompute scale factors (as buffers for efficient device transfer)
        stream_weights = 2.0 ** (torch.arange(self.input_streams) * self.input_bits_per_stream)
        slice_weights = 2.0 ** (torch.arange(self.weight_slices) * self.weight_bits_per_slice)
                
               


        self.register_buffer('stream_scale', stream_weights.view(1, 1, 1, -1, 1))
        self.register_buffer('slice_scale', slice_weights.view(1, 1, 1, 1, -1))
   

    def _fix_input_sign_bit(self, ps):
        """
        修正补码input的符号位
        
        补码的符号位(MSB)权重应该是负的，但MVM按正数算了
        解决：把包含符号位的stream取反
        
        Args:
            ps: [batch, out_ch, patches, streams, slices]
        Returns:
            修正后的ps（现在input是真正的有符号数）
        """
        if self.input_bits_per_stream != 1:
            print(f"⚠️ Warning: input bits_per_stream={self.input_bits_per_stream} != 1, "
                  f"sign handling may be inaccurate")
            return ps
        
        sign_stream_idx = self.input_bits - 1
        result = ps.clone()
        result[:, :, :, sign_stream_idx, :] = -ps[:, :, :, sign_stream_idx, :]
        return result
    
    def compute_vectorized_conv(self, inputs, weights):
            

        input_patches = F.unfold(inputs, self.kernel_size, self.dilation, self.padding, self.stride)
        batch_size, patch_features, num_patches = input_patches.shape
        input_streams  = VectorizedInputBitStreaming.apply(
             input_patches, self.input_bits, self.input_frac_bits,
             self.input_bits_per_stream, self.input_streams
         )
       
        # Bit slicing for weights
        pos_slices, neg_slices, norm_factor = VectorizedWeightBitSlicing.apply(
            weights, self.weight_bits,   self.weight_bits_per_slice
        )
        
        pos_weight_matrix = pos_slices.view(self.out_channels, -1, self.weight_slices)
        neg_weight_matrix = neg_slices.view(self.out_channels, -1, self.weight_slices)
        
        total_adc_loss = torch.tensor(0.0, device=inputs.device)
        
        if self.num_subarrays > 1:
            input_chunks = torch.chunk(input_streams, self.num_subarrays, dim=1)
            pos_chunks = torch.chunk(pos_weight_matrix, self.num_subarrays, dim=1)
            neg_chunks = torch.chunk(neg_weight_matrix, self.num_subarrays, dim=1)
            
            chunk_results = []
            for input_chunk, pos_chunk, neg_chunk in zip(input_chunks, pos_chunks, neg_chunks):
                pos_results = torch.einsum('bfps,oft->bopst', input_chunk, pos_chunk)
                neg_results = torch.einsum('bfps,oft->bopst', input_chunk, neg_chunk)
                #  # 🆕 修正符号位
                # pos_results = self._fix_input_sign_bit(pos_results)
                # neg_results = self._fix_input_sign_bit(neg_results) 
                
                pos_quantized, pos_loss = self.adc_pos(pos_results)
                neg_quantized, neg_loss = self.adc_neg(neg_results)               
                
                total_adc_loss = total_adc_loss + pos_loss + neg_loss
              
                pos_scaled = pos_quantized * self.stream_scale * self.slice_scale
                neg_scaled = neg_quantized * self.stream_scale * self.slice_scale
                
                chunk_result = (pos_scaled - neg_scaled).sum(dim=(-2, -1))
                chunk_results.append(chunk_result)
            
            final_output = torch.stack(chunk_results, dim=0).sum(dim=0)
            
        else:
            pos_results = torch.einsum('bfps,oft->bopst', input_streams, pos_weight_matrix)
            neg_results = torch.einsum('bfps,oft->bopst', input_streams, neg_weight_matrix)
            # #  # 🆕 修正符号位
            # pos_results = self._fix_input_sign_bit(pos_results)
            # neg_results = self._fix_input_sign_bit(neg_results)
            
            pos_quantized, pos_loss = self.adc_pos(pos_results)
            neg_quantized, neg_loss = self.adc_neg(neg_results)
            
            total_adc_loss = pos_loss + neg_loss
         
            pos_scaled = pos_quantized * self.stream_scale * self.slice_scale
            neg_scaled = neg_quantized * self.stream_scale * self.slice_scale
            
            final_output = (pos_scaled - neg_scaled).sum(dim=(-2, -1))
        
        
        final_output = final_output * norm_factor.view(1, -1, 1)   
        weight_quant_scale = (2 ** self.weight_bits) - 1
        final_output = final_output / weight_quant_scale
        
        input_max_int =  2 ** (self.input_bits) 
        final_output = final_output / input_max_int
       

        output_h = self._calc_output_size(inputs.shape[2], 0)
        output_w = self._calc_output_size(inputs.shape[3], 1)
        output = F.fold(final_output, (output_h, output_w), (1, 1))
        
        return output, total_adc_loss

    def _calc_output_size(self, input_size, dim):
        kernel = self.kernel_size
        pad = self.padding[dim]
        dilation = self.dilation[dim]
        stride = self.stride[dim]
        return (input_size + 2 * pad - dilation * (kernel - 1) - 1) // stride + 1
    
    def forward(self, inputs):
        if self.experiment_state == "PTQAT" and self.num_subarrays > 0:
           ## if self.weight_bits > 0 or self.input_bits > 0:
                output, adc_loss = self.compute_vectorized_conv(inputs, self.weight)
                return output, adc_loss
        elif self.experiment_state == "QAT":

                qa = InputQuantization.apply(inputs, self.input_bits, self.input_frac_bits)
                qw = WeightQuantization.apply(self.weight, self.weight_bits)
                output = F.conv2d(qa, qw , bias=None,
                              stride=self.stride, padding=self.padding,
                              dilation=self.dilation, groups=self.groups)
              
                return output, torch.tensor(0.0, device=inputs.device)
        elif self.experiment_state == "pretraining" or self.experiment_state == "pruning":
            output = F.conv2d(inputs, self.weight, bias=None,
                          stride=self.stride, padding=self.padding,
                          dilation=self.dilation, groups=self.groups)
            return output, torch.tensor(0.0, device=inputs.device)




















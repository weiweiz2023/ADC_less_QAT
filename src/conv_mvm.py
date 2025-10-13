import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from src.adc_module import Nbit_ADC

class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim):
        super().__init__()
        self.num_embeddings = num_embeddings  # 16 for 4-bit
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        # 初始化码本为均匀分布
        self.embedding.weight.data.uniform_(-1/num_embeddings, 1/num_embeddings)
    
    def forward(self, weights_flat):
        # 计算到所有码本的距离
        distances = torch.cdist(weights_flat, self.embedding.weight)
        
        # 找最近的码本索引
        indices = torch.argmin(distances, dim=1)
        
        # 量化
        quantized = F.embedding(indices, self.embedding.weight)
        
        # 计算VQ损失
        vq_loss = F.mse_loss(quantized.detach(), weights_flat) + \
                  0.1 * F.mse_loss(quantized, weights_flat.detach())
        
        # 直通估计器
        quantized = weights_flat + (quantized - weights_flat).detach()
        
        return quantized, vq_loss    
    
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
        max_abs = ctx.max_abs
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

        self.adc_slice_weighting = getattr(arch_args, 'adc_slice_weighting', 'exponential')
        self.adc_slice_weight_scale = getattr(arch_args, 'adc_slice_weight_scale', 1.0)
        
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
        
        # ADC modules
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
        #  VQ 
        if arch_args.use_vq:
            num_embeddings = 2 ** self.weight_bits  # 4位 = 16个码本
            self.vq_layer = VectorQuantizer(num_embeddings, embedding_dim=1)
        else:
            self.vq_layer = None
    def compute_vectorized_conv(self, inputs, weights):
        # Unfold input to patches
        input_patches = F.unfold(inputs, self.kernel_size, self.dilation, self.padding, self.stride)
        batch_size, patch_features, num_patches = input_patches.shape
         # 🆕 在位切片之前添加VQ
        vq_loss = torch.tensor(0.0, device=weights.device)
        
        if self.vq_layer is not None:
            # 展平权重
            weights_flat = weights.view(-1, 1)
            
            # VQ量化
            weights_quantized, vq_loss = self.vq_layer(weights_flat)
            
            # 恢复形状
            weights = weights_quantized.view(weights.shape)

        # Bit streaming for inputs
        input_streams = VectorizedInputBitStreaming.apply(
            input_patches, self.input_bits, self.input_frac_bits,
            self.input_bits_per_stream, self.input_streams
        )
        
        # Bit slicing for weights
        pos_slices, neg_slices, norm_factor = VectorizedWeightBitSlicing.apply(
            weights, self.weight_bits, self.weight_bits_per_slice
        )
        
        pos_weight_matrix = pos_slices.view(self.out_channels, -1, self.weight_slices)
        neg_weight_matrix = neg_slices.view(self.out_channels, -1, self.weight_slices)
        
        total_adc_loss = torch.tensor(0.0, device=inputs.device)
        
        if self.num_subarrays > 1:
            # Split into chunks
            input_chunks = torch.chunk(input_streams, self.num_subarrays, dim=1)
            pos_chunks = torch.chunk(pos_weight_matrix, self.num_subarrays, dim=1)
            neg_chunks = torch.chunk(neg_weight_matrix, self.num_subarrays, dim=1)
            
            # Process each chunk separately (IMPORTANT for 1-bit ADC!)
            chunk_results = []
            for input_chunk, pos_chunk, neg_chunk in zip(input_chunks, pos_chunks, neg_chunks):
                # MVM
                pos_res = torch.einsum('bfps,oft->bopst', input_chunk, pos_chunk)
                neg_res = torch.einsum('bfps,oft->bopst', input_chunk, neg_chunk)
                
                # ADC quantization per chunk (preserves sign information for 1-bit)
                pos_quantized, pos_loss = self.adc_pos(pos_res)
                neg_quantized, neg_loss = self.adc_neg(neg_res)
                
                total_adc_loss = total_adc_loss + pos_loss + neg_loss
                
                # Scale this chunk
                pos_scaled = pos_quantized * self.stream_scale * self.slice_scale
                neg_scaled = neg_quantized * self.stream_scale * self.slice_scale
                
                chunk_result = (pos_scaled - neg_scaled).sum(dim=(-2, -1))
                chunk_results.append(chunk_result)
            
            # Sum scaled results from all chunks
            final_output = torch.stack(chunk_results, dim=0).sum(dim=0)
            
        else:
            # No subarray splitting
            pos_results = torch.einsum('bfps,oft->bopst', input_streams, pos_weight_matrix)
            neg_results = torch.einsum('bfps,oft->bopst', input_streams, neg_weight_matrix)
            
            pos_quantized, pos_loss = self.adc_pos(pos_results)
            neg_quantized, neg_loss = self.adc_neg(neg_results)
            
            total_adc_loss = pos_loss + neg_loss
            
            pos_scaled = pos_quantized * self.stream_scale * self.slice_scale
            neg_scaled = neg_quantized * self.stream_scale * self.slice_scale
            
            final_output = (pos_scaled - neg_scaled).sum(dim=(-2, -1))
        
        # Apply normalization
        final_output = final_output * norm_factor
        input_scale = 2 ** self.input_bits - 1
        final_output = final_output / input_scale
        
        # Fold back to spatial dimensions
        output_h = self._calc_output_size(inputs.shape[2], 0)
        output_w = self._calc_output_size(inputs.shape[3], 1)
        output = F.fold(final_output, (output_h, output_w), (1, 1))
        
        return output, total_adc_loss + vq_loss

    def _calc_output_size(self, input_size, dim):
        kernel = self.kernel_size
        pad = self.padding[dim]
        dilation = self.dilation[dim]
        stride = self.stride[dim]
        return (input_size + 2 * pad - dilation * (kernel - 1) - 1) // stride + 1
    
    def forward(self, inputs):
        if self.experiment_state == "PTQAT" and self.num_subarrays > 0:
            if self.weight_bits > 0 or self.input_bits > 0:
                output, vq_loss = self.compute_vectorized_conv(inputs, self.weight)
                return output, vq_loss
            else:
                output = F.conv2d(inputs, self.weight, bias=None,
                              stride=self.stride, padding=self.padding,
                              dilation=self.dilation, groups=self.groups)
                return output, torch.tensor(0.0, device=inputs.device)
        else:
            output = F.conv2d(inputs, self.weight, bias=None,
                          stride=self.stride, padding=self.padding,
                          dilation=self.dilation, groups=self.groups)
            return output, torch.tensor(0.0, device=inputs.device)











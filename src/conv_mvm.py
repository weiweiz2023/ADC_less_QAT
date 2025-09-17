



import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost=0.25):
        super(VectorQuantizer, self).__init__()
        
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings  # 2^4 = 16 for 4-bit
        self.commitment_cost = commitment_cost
        
        # Initialize codebook
        self.embedding = nn.Embedding(self.num_embeddings, self.embedding_dim)
        self.embedding.weight.data.uniform_(-1/self.num_embeddings, 1/self.num_embeddings)
        
    def forward(self, inputs):
        input_shape = inputs.shape
        flat_input = inputs.view(-1, self.embedding_dim)
        
        distances = (torch.sum(flat_input**2, dim=1, keepdim=True) + 
                    torch.sum(self.embedding.weight**2, dim=1) - 
                    2 * torch.matmul(flat_input, self.embedding.weight.t()))
        
        encoding_indices = torch.argmin(distances, dim=1)
        quantized = F.embedding(encoding_indices, self.embedding.weight)
        quantized = quantized.view(input_shape)
        
        # 损失计算 - 注意尺度
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        
        # 关键：基于距离的自适应缩放
        with torch.no_grad():
            min_distances = torch.min(distances, dim=1)[0]
            avg_distance = torch.mean(min_distances)
            
            # 如果距离已经很小，降低损失权重
            if avg_distance < 0.001:
                loss_scale = 0.1  # 距离小时大幅降低损失
            elif avg_distance < 0.01:
                loss_scale = 0.5
            else:
                loss_scale = 1.0
        
        # 应用自适应缩放
        scaled_commitment_cost = self.commitment_cost * loss_scale
        vq_loss = q_latent_loss + scaled_commitment_cost * e_latent_loss
        
        # 诊断信息
        # with torch.no_grad():
        #     if torch.rand(1) < 0.001:
        #         print(f"VQ状态: 距离={avg_distance:.6f}, 缩放={loss_scale:.2f}, "
        #               f"原始损失={q_latent_loss + self.commitment_cost * e_latent_loss:.6f}, "
        #               f"缩放后={vq_loss:.6f}")
        
        quantized = inputs + (quantized - inputs).detach()
        return quantized, vq_loss, encoding_indices.view(input_shape[:-1])

class VQWeightBitSlicing(Function):
    @staticmethod
    def forward(ctx, weights, weight_bits, bits_per_slice, vq_layer,use_vq=True):
        # print(f"VQWeightBitSlicing DEBUG:")
        # print(f"  use_vq: {use_vq}")
        # print(f"  vq_layer: {vq_layer}")
        # print(f"  weights shape: {weights.shape}")
        # print(f"  weights range: [{weights.min():.4f}, {weights.max():.4f}]")
        ctx.save_for_backward(weights)
        ctx.weight_bits = weight_bits
        ctx.vq_layer = vq_layer
        
        # Step 1: Apply VQ to weights
        original_shape = weights.shape
        # Reshape for VQ (treat each weight as 1D embedding)
        weights_reshaped = weights.view(-1, 1)
        # print(f"  weights_reshaped shape: {weights_reshaped.shape}")
        if use_vq:
            # print("  正在应用VQ...")
            vq_weights, vq_loss_computed, vq_indices = vq_layer(weights_reshaped)
            weights = vq_weights.view(original_shape)
            vq_loss = vq_loss_computed
            # print(f"  VQ应用成功!")
            # print(f"  VQ loss: {vq_loss}")
            # print(f"  量化前后权重变化: {torch.mean(torch.abs(weights_reshaped.view(-1) - weights.view(-1)))}")
        else:
            vq_loss=0
            weights= weights
        if not isinstance(vq_loss, torch.Tensor):
            vq_loss = torch.tensor(float(vq_loss), device=weights.device)
    
        # print(f"  最终VQ loss: {vq_loss} (type: {type(vq_loss)})")
        # Step 2: Separate positive and negative components of VQ weights
        weights_pos = torch.clamp(weights, min=0.0)
        weights_neg = torch.clamp(-weights, min=0.0)
        
        # Step 3: Normalize by max absolute value
        max_abs = max(weights_pos.max(), weights_neg.max())
        if max_abs > 0:
            weights_pos = weights_pos / max_abs
            weights_neg = weights_neg / max_abs
        else:
            max_abs = 1.0
        
        # Step 4: Convert VQ indices to bit slices directly
        # VQ indices are already in range [0, 2^weight_bits - 1]
        #vq_indices_reshaped = vq_indices.view(original_shape)
        
        num_slices = weight_bits // bits_per_slice
        bit_positions = torch.arange(0, weight_bits, bits_per_slice, device=weights.device)
        
        # Create bit slices from VQ indices
       # pos_slices = (vq_indices_reshaped.unsqueeze(-1) >> bit_positions) & ((1 << bits_per_slice) - 1)
       # neg_slices = torch.zeros_like(pos_slices)  # VQ indices are unsigned, so neg is zero
        
        # Alternative approach: bit slice the normalized VQ weights
        # This might give better results than using indices directly
        max_val = (2 ** weight_bits) - 1
        pos_int = torch.round(weights_pos * max_val).int()
        neg_int = torch.round(weights_neg * max_val).int()
        
        pos_slices_alt = (pos_int.unsqueeze(-1) >> bit_positions) & ((1 << bits_per_slice) - 1)
        neg_slices_alt = (neg_int.unsqueeze(-1) >> bit_positions) & ((1 << bits_per_slice) - 1)
        #print(f"  返回VQ loss: {vq_loss}")
        return pos_slices_alt.float(), neg_slices_alt.float(), max_abs, vq_loss
    
    @staticmethod
    def backward(ctx, grad_pos, grad_neg, grad_norm, grad_vq_loss):
        weights, = ctx.saved_tensors
        
        # Combine gradients from positive and negative slices
        grad_input = torch.where(weights >= 0, grad_pos.mean(dim=-1),-grad_neg.mean(dim=-1))
        grad_input = torch.clamp(grad_input, -1.0, 1.0)
        
        return grad_input, None, None, None, None
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

# Modified quantized_conv class
class quantized_conv(nn.Module):
    def __init__(self, in_channels, out_channels, arch_args, 
                 kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=None):
        super(quantized_conv, self).__init__()
        
        # Copy all the existing initialization from your quantized_conv
        self.weight_bits = arch_args.weight_bits
        self.weight_bits_per_slice = arch_args.bit_slice
        self.weight_slices = max(1, self.weight_bits // max(self.weight_bits_per_slice, 1))
        self.weight_frac_bits = arch_args.weight_bit_frac
        
        self.input_bits = arch_args.input_bits
        self.input_bits_per_stream = arch_args.bit_stream
        self.input_streams = max(1, self.input_bits // max(self.input_bits_per_stream, 1))
        self.input_frac_bits = arch_args.input_bit_frac
        
        
        self.use_vq = arch_args.use_vq
        self.commitment_cost = arch_args.commitment_cost


        # ADC parameters
        self.adc_bits = arch_args.adc_bit
        self.adc_grad_filter = arch_args.adc_grad_filter
        self.save_adc_data = arch_args.save_adc
        self.adc_custom_loss = arch_args.adc_custom_loss
        self.adc_reg_lambda = arch_args.adc_reg_lambda
        # Conv parameters
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = (stride, stride)
        self.padding = (padding, padding)
        self.dilation = (dilation, dilation)
        self.groups = groups
        
        # Subarray configuration
        subarray_size = arch_args.subarray_size
        if subarray_size <= 0:
            self.num_subarrays = 0
        else:
            total_inputs = in_channels * (kernel_size ** 2)
            self.num_subarrays = max(1, (total_inputs + subarray_size - 1) // subarray_size)
        
        self.experiment_state = arch_args.experiment_state
        
        # Weight parameter
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, kernel_size, kernel_size))
        nn.init.kaiming_normal_(self.weight)
        
        if self.use_vq:
            num_embeddings = 2 ** self.weight_bits  # 16 for 4-bit
            commitment_cost =self.commitment_cost #getattr(arch_args, 'vq_commitment_cost', 0.25)
            self.vq_layer = VectorQuantizer(num_embeddings, embedding_dim=1, 
                                          commitment_cost=commitment_cost)
            print(f"VQ已启用: {num_embeddings}个码本条目, commitment_cost={commitment_cost}")
        else:
            self.vq_layer = None
            print("使用原始量化方法")
        
        # ADC modules (keep your existing ADC setup)
        from src.adc_module import Nbit_ADC
        self.adc_pos = Nbit_ADC(self.adc_bits, self.weight_slices, self.input_streams, 
                               self.save_adc_data, self.adc_grad_filter,self.adc_custom_loss,self.adc_reg_lambda)
        self.adc_neg = Nbit_ADC(self.adc_bits, self.weight_slices, self.input_streams, 
                               self.save_adc_data, self.adc_grad_filter,self.adc_custom_loss,self.adc_reg_lambda)
    
    def compute_vectorized_conv_vq(self, inputs, weights):
        # print(f"DEBUG: use_vq = {self.use_vq}")
        # print(f"DEBUG: vq_layer = {self.vq_layer}")
        # Input processing (same as before)
        input_patches = F.unfold(inputs, self.kernel_size, self.dilation, self.padding, self.stride)
        batch_size, patch_features, num_patches = input_patches.shape
        
      
        from src.conv_mvm import VectorizedInputBitStreaming
        input_streams = VectorizedInputBitStreaming.apply(
            input_patches, self.input_bits, self.input_frac_bits,
            self.input_bits_per_stream, self.input_streams
        )
        
        pos_slices, neg_slices, norm_factor, vq_loss = VQWeightBitSlicing.apply(
            weights, self.weight_bits, self.weight_bits_per_slice, self.vq_layer,self.use_vq
        )
        # 添加VQ损失调试
        # print(f"DEBUG: VQ loss from bit slicing = {vq_loss}")
        # print(f"DEBUG: VQ loss type = {type(vq_loss)}")
        
        pos_weight_matrix = pos_slices.view(self.out_channels, -1, self.weight_slices)
        neg_weight_matrix = neg_slices.view(self.out_channels, -1, self.weight_slices)

        total_adc_loss = torch.tensor(0.0, device=inputs.device)  # 初始化总ADC损失
        # Rest of the computation (same as your existing implementation)
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
                total_adc_loss = total_adc_loss + chunk_loss
            final_output = torch.stack(results, dim=0).sum(dim=0)
        else:
            final_output, total_adc_loss  = self._process_subarray_vectorized(
                input_streams, pos_weight_matrix, neg_weight_matrix, batch_size
            )
        
        # Apply normalization and scaling
        final_output = final_output * norm_factor
        input_scale = 2 ** self.input_bits - 1
        final_output = final_output / input_scale
        
        output_h = self._calc_output_size(inputs.shape[2], 0)
        output_w = self._calc_output_size(inputs.shape[3], 1)
        output = F.fold(final_output, (output_h, output_w), (1, 1))
        #print(f"DEBUG: Final VQ loss = {vq_loss}")
        return output, vq_loss, total_adc_loss 
    
    def _process_subarray_vectorized(self, input_chunk, weight_pos_chunk, weight_neg_chunk, batch_size):
        # Same implementation as your existing method
        device = input_chunk.device
        stream_weights = 2.0 ** (torch.arange(self.input_streams, device=device) * self.input_bits_per_stream)
        slice_weights = 2.0 ** (torch.arange(self.weight_slices, device=device) * self.weight_bits_per_slice)
        
        pos_results = torch.einsum('bfps,oft->bopst', input_chunk, weight_pos_chunk)
        neg_results = torch.einsum('bfps,oft->bopst', input_chunk, weight_neg_chunk)
        
        if torch.any(neg_results.abs() > 100):
            pos_results = torch.clamp(pos_results, -50, 50)
            neg_results = torch.clamp(neg_results, -50, 50)

        original_shape = pos_results.shape
        pos_flat = pos_results.flatten()
        neg_flat = neg_results.flatten()

        pos_quantized, pos_loss = self.adc_pos(pos_flat)
        neg_quantized, neg_loss = self.adc_neg(neg_flat)
        
        pos_quantized = pos_quantized.view(original_shape)
        neg_quantized = neg_quantized.view(original_shape)

        total_adc_loss = pos_loss + neg_loss

        stream_weights = stream_weights.to(device).view(1, 1, 1, -1, 1)
        slice_weights = slice_weights.to(device).view(1, 1, 1, 1, -1)
        
        pos_scaled = pos_quantized * stream_weights * slice_weights
        neg_scaled = neg_quantized * stream_weights * slice_weights
        
        pos_final = pos_scaled.sum(dim=(-2, -1))
        neg_final = neg_scaled.sum(dim=(-2, -1))
        
        return pos_final - neg_final, total_adc_loss
    
    def _calc_output_size(self, input_size, dim):
        kernel = self.kernel_size
        pad = self.padding[dim]
        dilation = self.dilation[dim]
        stride = self.stride[dim]
        return (input_size + 2 * pad - dilation * (kernel - 1) - 1) // stride + 1
    
    def forward(self, inputs):
        if self.experiment_state == "PTQAT" and self.num_subarrays > 0:
            if self.weight_bits > 0 or self.input_bits > 0:
                output, vq_loss, total_adc_loss  = self.compute_vectorized_conv_vq(inputs, self.weight)
                return output, vq_loss, total_adc_loss   # Return VQ loss to be added to total loss
            else:
                return F.conv2d(inputs, self.weight, bias=None,
                              stride=self.stride, padding=self.padding,
                              dilation=self.dilation, groups=self.groups), 0
        else:
            return F.conv2d(inputs, self.weight, bias=None,
                          stride=self.stride, padding=self.padding,
                          dilation=self.dilation, groups=self.groups), 0

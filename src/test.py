import torch
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict



class QuantizationComparator:
    def __init__(self, model):
        self.model = model
        self.comparison_data = defaultdict(dict)
        
    def uniform_quantization(self, weights, bits=4):
        """标准均匀量化"""
        # 计算量化范围
        max_val = torch.max(torch.abs(weights))
        min_val = -max_val
        
        # 计算量化步长
        num_levels = 2 ** bits
        scale = (max_val - min_val) / (num_levels - 1)
        zero_point = 0
        
        # 量化
        quantized = torch.round((weights - zero_point) / scale) * scale + zero_point
        quantized = torch.clamp(quantized, min_val, max_val)
        
        return quantized, scale
    
    def vq_quantization(self, weights, vq_layer):
        """VQ量化"""
        if vq_layer is None:
            return weights, 0
            
        original_shape = weights.shape
        weights_flat = weights.view(-1, 1)
        quantized_flat, vq_loss, indices = vq_layer(weights_flat)
        quantized = quantized_flat.view(original_shape)
        
        return quantized, vq_loss.item()
    
    def analyze_layer_quantization(self, layer_name, weights, vq_layer=None):
        """分析单层的量化效果"""
        original_weights = weights.detach().clone()
        
        # 普通量化
        uniform_quantized, uniform_scale = self.uniform_quantization(original_weights)
        
        # VQ量化
        vq_quantized, vq_loss = self.vq_quantization(original_weights, vq_layer)
        
        # 计算误差指标
        uniform_mse = torch.mean((original_weights - uniform_quantized) ** 2).item()
        vq_mse = torch.mean((original_weights - vq_quantized) ** 2).item()
        
        uniform_mae = torch.mean(torch.abs(original_weights - uniform_quantized)).item()
        vq_mae = torch.mean(torch.abs(original_weights - vq_quantized)).item()
        
        # 计算信噪比
        signal_power = torch.mean(original_weights ** 2).item()
        uniform_snr = 10 * np.log10(signal_power / (uniform_mse + 1e-10))
        vq_snr = 10 * np.log10(signal_power / (vq_mse + 1e-10))
        
        # 统计信息
        stats = {
            'layer_name': layer_name,
            'original_stats': {
                'mean': original_weights.mean().item(),
                'std': original_weights.std().item(),
                'min': original_weights.min().item(),
                'max': original_weights.max().item(),
                'num_params': original_weights.numel()
            },
            'uniform_quantization': {
                'mse': uniform_mse,
                'mae': uniform_mae,
                'snr_db': uniform_snr,
                'scale': uniform_scale,
                'quantized_range': [uniform_quantized.min().item(), uniform_quantized.max().item()]
            },
            'vq_quantization': {
                'mse': vq_mse,
                'mae': vq_mae,
                'snr_db': vq_snr,
                'vq_loss': vq_loss,
                'quantized_range': [vq_quantized.min().item(), vq_quantized.max().item()]
            },
            'comparison': {
                'mse_improvement': ((uniform_mse - vq_mse) / uniform_mse * 100) if uniform_mse > 0 else 0,
                'mae_improvement': ((uniform_mae - vq_mae) / uniform_mae * 100) if uniform_mae > 0 else 0,
                'snr_improvement': vq_snr - uniform_snr
            }
        }
        
        # 保存量化后的权重用于可视化
        stats['weights'] = {
            'original': original_weights.cpu().numpy().flatten(),
            'uniform': uniform_quantized.cpu().numpy().flatten(),
            'vq': vq_quantized.cpu().numpy().flatten()
        }
        
        self.comparison_data[layer_name] = stats
        return stats
    
    def analyze_full_model(self):
        """分析整个模型的量化效果"""
        print("=" * 60)
        print("模型量化效果全面分析")
        print("=" * 60)
        
        total_params = 0
        uniform_total_error = 0
        vq_total_error = 0
        
        for name, module in self.model.named_modules():
            if hasattr(module, 'weight') and hasattr(module, 'vq_layer'):
                vq_layer = getattr(module, 'vq_layer', None)
                stats = self.analyze_layer_quantization(name, module.weight.data, vq_layer)
                
                # 累积统计
                num_params = stats['original_stats']['num_params']
                total_params += num_params
                uniform_total_error += stats['uniform_quantization']['mse'] * num_params
                vq_total_error += stats['vq_quantization']['mse'] * num_params
                
                # 打印层级统计
                print(f"\n层: {name}")
                print(f"  参数数量: {num_params:,}")
                print(f"  原始权重范围: [{stats['original_stats']['min']:.4f}, {stats['original_stats']['max']:.4f}]")
                print(f"  普通量化 MSE: {stats['uniform_quantization']['mse']:.6f}")
                print(f"  VQ量化 MSE: {stats['vq_quantization']['mse']:.6f}")
                print(f"  MSE改善: {stats['comparison']['mse_improvement']:.2f}%")
                print(f"  SNR改善: {stats['comparison']['snr_improvement']:.2f} dB")
        
        # 全局统计
        if total_params > 0:
            avg_uniform_mse = uniform_total_error / total_params
            avg_vq_mse = vq_total_error / total_params
            global_improvement = ((avg_uniform_mse - avg_vq_mse) / avg_uniform_mse * 100)
            
            print(f"\n" + "=" * 40)
            print(f"全局统计:")
            print(f"  总参数: {total_params:,}")
            print(f"  平均普通量化MSE: {avg_uniform_mse:.6f}")
            print(f"  平均VQ量化MSE: {avg_vq_mse:.6f}")
            print(f"  全局MSE改善: {global_improvement:.2f}%")
            print("=" * 40)
    
    def visualize_quantization_comparison(self, layer_name, save_path=None):
        """可视化特定层的量化对比"""
        if layer_name not in self.comparison_data:
            print(f"层 {layer_name} 的数据不存在")
            return
        
        data = self.comparison_data[layer_name]
        weights = data['weights']
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'量化对比分析 - {layer_name}', fontsize=16)
        
        # 1. 权重分布直方图
        axes[0, 0].hist(weights['original'], bins=50, alpha=0.7, label='原始权重', density=True)
        axes[0, 0].hist(weights['uniform'], bins=50, alpha=0.7, label='普通量化', density=True)
        axes[0, 0].hist(weights['vq'], bins=50, alpha=0.7, label='VQ量化', density=True)
        axes[0, 0].set_title('权重分布对比')
        axes[0, 0].set_xlabel('权重值')
        axes[0, 0].set_ylabel('密度')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. 量化误差分布
        uniform_error = weights['original'] - weights['uniform']
        vq_error = weights['original'] - weights['vq']
        
        axes[0, 1].hist(uniform_error, bins=50, alpha=0.7, label=f'普通量化误差 (std={np.std(uniform_error):.4f})')
        axes[0, 1].hist(vq_error, bins=50, alpha=0.7, label=f'VQ量化误差 (std={np.std(vq_error):.4f})')
        axes[0, 1].set_title('量化误差分布')
        axes[0, 1].set_xlabel('误差值')
        axes[0, 1].set_ylabel('频次')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. 误差散点图
        sample_indices = np.random.choice(len(weights['original']), min(1000, len(weights['original'])), replace=False)
        axes[1, 0].scatter(uniform_error[sample_indices], vq_error[sample_indices], alpha=0.6, s=1)
        axes[1, 0].plot([-0.1, 0.1], [-0.1, 0.1], 'r--', label='y=x')
        axes[1, 0].set_title('误差对比散点图')
        axes[1, 0].set_xlabel('普通量化误差')
        axes[1, 0].set_ylabel('VQ量化误差')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. 量化指标对比
        metrics = ['MSE', 'MAE', 'SNR (dB)']
        uniform_values = [
            data['uniform_quantization']['mse'],
            data['uniform_quantization']['mae'],
            data['uniform_quantization']['snr_db']
        ]
        vq_values = [
            data['vq_quantization']['mse'],
            data['vq_quantization']['mae'],
            data['vq_quantization']['snr_db']
        ]
        
        x = np.arange(len(metrics))
        width = 0.35
        
        axes[1, 1].bar(x - width/2, uniform_values, width, label='普通量化', alpha=0.8)
        axes[1, 1].bar(x + width/2, vq_values, width, label='VQ量化', alpha=0.8)
        
        axes[1, 1].set_title('量化指标对比')
        axes[1, 1].set_xlabel('指标')
        axes[1, 1].set_ylabel('数值')
        axes[1, 1].set_xticks(x)
        axes[1, 1].set_xticklabels(metrics)
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        # 添加改善百分比注释
        for i, (uniform_val, vq_val) in enumerate(zip(uniform_values, vq_values)):
            if uniform_val != 0:
                improvement = ((uniform_val - vq_val) / uniform_val * 100)
                axes[1, 1].text(i, max(uniform_val, vq_val) * 1.1, f'{improvement:+.1f}%', 
                               ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"图表已保存到: {save_path}")
        
        plt.show()
    
    def generate_comparison_report(self):
        """生成详细的对比报告"""
        print("\n" + "=" * 80)
        print("量化方法对比报告")
        print("=" * 80)
        
        if not self.comparison_data:
            print("没有分析数据，请先运行 analyze_full_model()")
            return
        
        # 按改善程度排序
        layers_by_improvement = sorted(
            self.comparison_data.items(),
            key=lambda x: x[1]['comparison']['mse_improvement'],
            reverse=True
        )
        
        print(f"\n按MSE改善程度排序的层级分析:")
        print("-" * 80)
        print(f"{'层名':<30} {'MSE改善%':<10} {'MAE改善%':<10} {'SNR改善dB':<12} {'参数数':<10}")
        print("-" * 80)
        
        total_improved = 0
        total_degraded = 0
        
        for layer_name, stats in layers_by_improvement:
            mse_imp = stats['comparison']['mse_improvement']
            mae_imp = stats['comparison']['mae_improvement']
            snr_imp = stats['comparison']['snr_improvement']
            num_params = stats['original_stats']['num_params']
            
            print(f"{layer_name:<30} {mse_imp:>8.2f}   {mae_imp:>8.2f}   {snr_imp:>10.2f}   {num_params:>8,}")
            
            if mse_imp > 0:
                total_improved += 1
            else:
                total_degraded += 1
        
        print("-" * 80)
        print(f"总结: {total_improved} 层改善, {total_degraded} 层退化")
        
        # 建议
        print(f"\n建议:")
        if total_improved > total_degraded:
            print("✓ VQ量化总体上优于普通量化")
        elif total_improved == total_degraded:
            print("→ VQ量化与普通量化效果相当")
        else:
            print("✗ VQ量化总体上不如普通量化")
        
        # 找出最有效和最无效的层
        best_layer = max(self.comparison_data.items(), key=lambda x: x[1]['comparison']['mse_improvement'])
        worst_layer = min(self.comparison_data.items(), key=lambda x: x[1]['comparison']['mse_improvement'])
        
        print(f"\nVQ最有效的层: {best_layer[0]} (MSE改善 {best_layer[1]['comparison']['mse_improvement']:.2f}%)")
        print(f"VQ最无效的层: {worst_layer[0]} (MSE变化 {worst_layer[1]['comparison']['mse_improvement']:.2f}%)")

# 使用示例
def run_quantization_comparison(model):
    """运行完整的量化对比分析"""
    comparator = QuantizationComparator(model)
    
    # 分析全模型
    comparator.analyze_full_model()
    
    # 生成报告
    comparator.generate_comparison_report()
    
    # 可视化几个关键层（如果需要）
    # comparator.visualize_quantization_comparison('layer1.0.conv1', 'quantization_comparison.png')
    
    return comparator
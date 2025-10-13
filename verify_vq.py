# verify_vq.py
import torch
import matplotlib.pyplot as plt
import numpy as np
import os

def visualize_vq_codebook(model, save_path='./saved/vq_codebook_visualization.png'):
    """Visualize VQ codebook and weight distribution"""
    
    # 收集所有VQ层
    vq_layers = []
    for name, module in model.named_modules():
        if hasattr(module, 'vq_layer') and module.vq_layer is not None:
            vq_layers.append((name, module))
    
    if not vq_layers:
        print("Warning: No VQ layers found!")
        return
    
    print(f"Found {len(vq_layers)} VQ layers, displaying first 6")
    
    num_display = min(6, len(vq_layers))
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    # 🆕 改为英文
    fig.suptitle('VQ Codebook Analysis - Red lines show codebook positions', 
                 fontsize=16, fontweight='bold')
    axes = axes.flatten()
    
    for idx in range(num_display):
        name, module = vq_layers[idx]
        ax = axes[idx]
        
        weights = module.weight.data.flatten().cpu().numpy()
        codebook = module.vq_layer.embedding.weight.data.squeeze().cpu().numpy()
        
        print(f"  Plotting {name}: weights={len(weights)}, codebook={len(codebook)}")
        
        # 绘制权重分布直方图
        counts, bins, patches = ax.hist(weights, bins=60, alpha=0.7, 
                                        label='Weight Distribution',  # 🆕 英文
                                        density=True, 
                                        color='skyblue', edgecolor='navy')
        
        # 标记码本位置
        y_max = counts.max() * 1.1 if counts.max() > 0 else 1
        for i, code in enumerate(codebook):
            ax.axvline(code, color='red', linestyle='--', 
                      alpha=0.8, linewidth=2, 
                      label='Codebook' if i == 0 else '')  # 🆕 英文
        
        # 美化
        short_name = '.'.join(name.split('.')[-2:]) if '.' in name else name
        # 🆕 英文标题
        ax.set_title(f'{short_name}\n{len(codebook)} codebooks | Range[{weights.min():.3f}, {weights.max():.3f}]', 
                    fontsize=11, fontweight='bold')
        ax.set_xlabel('Weight Value', fontsize=10)  # 🆕 英文
        ax.set_ylabel('Density', fontsize=10)  # 🆕 英文
        ax.legend(fontsize=9, loc='upper right')
        ax.grid(True, alpha=0.3, linestyle=':')
        ax.set_ylim(0, y_max)
    
    # 隐藏多余子图
    for idx in range(num_display, 6):
        axes[idx].axis('off')
    
    plt.tight_layout()
    
    # 保存
    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nVisualization saved: {save_path}")
    plt.close()
    
    return save_path


# 如果直接运行这个文件（用于独立可视化checkpoint）
if __name__ == '__main__':
    print("独立运行模式：加载checkpoint并可视化VQ码本")
    print("=" * 60)
    
    checkpoint_path = input("请输入checkpoint路径: ").strip()
    
    if not os.path.isfile(checkpoint_path):
        print(f"❌ 文件不存在: {checkpoint_path}")
        exit(1)
    
    print(f"加载 {checkpoint_path}...")
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        
        # 从checkpoint重建模型
        args = checkpoint.get('args', None)
        if args is None:
            print("❌ checkpoint中没有args信息，无法重建模型")
            exit(1)
        
        # 动态导入（需要在项目根目录运行）
        import sys
        sys.path.insert(0, os.path.dirname(__file__))
        
        from src.resnet import ResNet
        
        # 构建模型
        if args.model == "resnet20":
            num_blocks = [3, 3, 3]
            start_chan = 16
        else:
            print(f"❌ 不支持的模型: {args.model}")
            exit(1)
        
        in_channels = 1 if args.dataset == 'MNIST' else 3
        model = ResNet(num_blocks, in_channels, args, start_chan)
        
        # 加载权重
        model.load_state_dict(checkpoint['state_dict'], strict=False)
        print("✓ 模型加载成功")
        
        # 可视化
        visualize_vq_codebook(model)
        
    except Exception as e:
        print(f"❌ 出错: {e}")
        import traceback
        traceback.print_exc()
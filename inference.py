import argparse
import torch
import pyvista as pv
from data_loader import load_single_vti_as_tensor
from pipeline import FlowVortexPipeline

def main():
    parser = argparse.ArgumentParser(description="Run Flow Vortex Net on a single .vti file")
    parser.add_argument("vti_file", type=str, help="Path to the input .vti file")
    parser.add_argument("--save_mesh", action="store_true", help="Save output variables back to a new VTI file")
    parser.add_argument("--checkpoint", type=str, default="./checkpoints/mae_best_checkpoint.pth", help="Path to pre-trained MAE checkpoint")
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 1. 加载数据
    print(f"Loading .vti file: {args.vti_file}")
    try:
        # 默认使用 ("u", "v", "w") 作为速度分量名称，如果不匹配请调整
        tensor_input = load_single_vti_as_tensor(args.vti_file).to(device)
    except Exception as e:
        print(f"Error loading VTI file: {e}")
        print("提示: 确保输入是一组合法的结构化网格张量")
        return
        
    # 添加一个假的 Batch 维度
    if len(tensor_input.shape) == 4:
        tensor_input = tensor_input.unsqueeze(0)
        
    print(f"Input tensor shape: {tensor_input.shape} (B, C, D, H, W)")
    
    # 调整输入大小为可以被模型 patch_size 整除的形状。通常使用插值法
    # 这里我们为了演示直接传入，如果尺寸报错，您可能需要 Resize 或 Padding

    # 2. 初始化模型
    print("Initializing Model Pipeline for Inference...")
    # 当在下游推理时，我们依然使用 MAE 但将 mask_ratio 设为 0 (完全可见，纯特征提取和重建)
    import os
    use_mae = os.path.exists(args.checkpoint)
    
    pipeline = FlowVortexPipeline(
        use_mae=use_mae,
        patch_size=(2, 4, 4),
        in_chans=3,
        embed_dim=48, 
        depths=[2, 2, 6, 2], 
        num_heads=[3, 6, 12, 24], 
        window_size=(4, 4, 4),
        mask_ratio=0.0 # 推理阶段不屏蔽任何像素！
    )
    
    if use_mae:
        print(f"Loading pre-trained checkpoint from: {args.checkpoint}")
        try:
            checkpoint = torch.load(args.checkpoint, map_location=device)
            pipeline.load_state_dict(checkpoint['model_state_dict'])
            print("  -> Checkpoint loaded successfully!")
        except Exception as e:
            print(f"  -> Failed to load checkpoint: {e}")
            
    pipeline = pipeline.to(device)
    pipeline.eval()
    
    # 3. 前向计算
    print("Running inference...")
    with torch.no_grad():
        # dx, dy, dz 可根据实际物理网格间距修改
        velocity_field, mask, v_sol, v_irr, q_field = pipeline(tensor_input, dx=1.0, dy=1.0, dz=1.0)
    
    print("\n[Outputs generated]")
    print(f"Solenoidal flow (无散流场) shape : {v_sol.shape}")
    print(f"Irrotational flow (无旋流场) shape : {v_irr.shape}")
    print(f"Q-criterion field (Q准则涡特征) shape : {q_field.shape}")
    
    # 4. 可选：将结果写回 VTI，方便通过 ParaView 观察
    if args.save_mesh:
        mesh = pv.read(args.vti_file)
        
        # 将 PyTorch 张量转回 Numpy (并展平回 point data 或 cell data 格式)
        # 注意: v_sol 的 shape 为 (1, 3, D, H, W)
        v_sol_np = v_sol.squeeze(0).cpu().numpy() # (3, D, H, W)
        q_np = q_field.squeeze(0).cpu().numpy() # (D, H, W)
        
        dims = mesh.dimensions
        # VTK 数组恢复 (D, H, W) -> (N, 3) 按照 VTK 原生展平顺序 (x-fastest)
        u_sol = v_sol_np[0].flatten(order='C')
        v_sol_y = v_sol_np[1].flatten(order='C')
        w_sol = v_sol_np[2].flatten(order='C')
        
        q_flat = q_np.flatten(order='C')
        
        from numpy import stack
        vec_sol = stack([u_sol, v_sol_y, w_sol], axis=1)
        
        mesh.point_data["Velocity_Solenoidal"] = vec_sol
        mesh.point_data["Q_Criterion"] = q_flat
        
        out_name = args.vti_file.replace(".vti", "_features.vti")
        mesh.save(out_name)
        print(f"\nSaved features to: {out_name}")
        print("您现在可以使用 ParaView 打开它以进行三维可视化。")

if __name__ == "__main__":
    main()

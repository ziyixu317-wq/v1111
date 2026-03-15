import argparse
import torch
import pyvista as pv
from data_loader import load_single_vti_as_tensor
from pipeline import FlowVortexPipeline

def main():
    parser = argparse.ArgumentParser(description="Run Flow Vortex Net on a single .vti file")
    parser.add_argument("vti_file", type=str, help="Path to the input .vti file")
    parser.add_argument("--save_mesh", action="store_true", help="Save output variables back to a new VTI file")
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
        print("提示: 请检查 .vti 文件中是否包含纯量 u, v, w。如果速度场被存储为了单个 Vector 数组 (如 'velocity')，请修改 data_loader 调用方式。")
        return
        
    print(f"Input tensor shape: {tensor_input.shape} (B, C, D, H, W)")
    
    # 调整输入大小为可以被模型 patch_size 整除的形状。通常使用插值法
    # 这里我们为了演示直接传入，如果尺寸报错，您可能需要 Resize 或 Padding

    # 2. 初始化模型
    print("Initializing Model Pipeline...")
    # 由于真实的 VTI 尺寸可能很大，这里我们仅为了演示通过前向传播
    # 注意：如果显存不足或尺寸不被 swin transformer 支持，请在这里加入对 tensor_input 的插值/降采样
    pipeline = FlowVortexPipeline(use_mae=False) # 如果只是为了提取特征(解耦和Q准则)，我们可以关闭基于MAE的重建来节省显存和计算
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

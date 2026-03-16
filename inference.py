
import argparse
import torch
import pyvista as pv
import numpy as np
import os
from data_loader import load_single_vti_as_tensor
from pipeline import FlowVortexFusionPipeline
from vortex import calculate_ivd

def main():
    parser = argparse.ArgumentParser(description="Inference for Fused FlowVortexMAE")
    parser.add_argument("vti_file", type=str)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--mode", type=str, default='segmentation', choices=['pretrain', 'segmentation'])
    parser.add_argument("--save_dir", type=str, default="./inference_results")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. Load Data
    tensor_input = load_single_vti_as_tensor(args.vti_file).to(device)
    if len(tensor_input.shape) == 4:
        tensor_input = tensor_input.unsqueeze(0)
    
    # 2. Model
    pipeline = FlowVortexFusionPipeline(mode=args.mode, in_chans=tensor_input.shape[1])
    ckpt = torch.load(args.checkpoint, map_location=device)
    pipeline.load_state_dict(ckpt['model_state_dict'])
    pipeline.to(device)
    pipeline.eval()
    
    # 3. Inference
    with torch.no_grad():
        if args.mode == 'pretrain':
            x_rec, mask, ivd_pred = pipeline(tensor_input)
            # Reconstructed Flow
            out_v = x_rec.squeeze(0).cpu().numpy()
            out_ivd = ivd_pred.squeeze(0).cpu().numpy()
        else:
            seg_mask = pipeline(tensor_input)
            out_ivd = seg_mask.squeeze(0).squeeze(0).cpu().numpy()
            out_v = None

    # 4. Save
    mesh = pv.read(args.vti_file)
    if out_v is not None:
        u, v, w = out_v[0].flatten(order='C'), out_v[1].flatten(order='C'), out_v[2].flatten(order='C')
        mesh.point_data["Reconstructed_Velocity"] = np.stack([u, v, w], axis=1)
    
    mesh.point_data["Pred_Vortex_Field"] = out_ivd.flatten(order='C')
    
    # GT for comparison
    gt_ivd = calculate_ivd(tensor_input).squeeze(0).cpu().numpy()
    mesh.point_data["GT_IVD_Field"] = gt_ivd.flatten(order='C')
    
    out_path = os.path.join(args.save_dir, os.path.basename(args.vti_file).replace(".vti", "_fused.vti"))
    mesh.save(out_path)
    print(f"Results saved to {out_path}")

if __name__ == "__main__":
    main()

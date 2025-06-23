
import os
import numpy as np
import torch
import imageio.v2 as imageio  # Use v2 to avoid deprecation warning
from pathlib import Path
from tqdm import tqdm
from argparse import ArgumentParser

from src.utils.io_utils import load_config
from src.utils.utils import get_render_settings, render_gaussian_model
from src.entities.gaussian_model import GaussianModel
from src.entities.arguments import OptimizationParams

def load_camera_poses(ckpt_dir):
    return {
        "kf_ids": torch.load(ckpt_dir / "kf_ids.ckpt"),
        "estimated_kf_c2w": torch.load(ckpt_dir / "estimated_kf_c2w.ckpt")
    }

def render_and_save_images(gaussian_model_path, poses, dataset_path, intrinsics, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    kf_ids = poses["kf_ids"].int().tolist()
    c2ws = poses["estimated_kf_c2w"]

    H, W = int(intrinsics["H"]), int(intrinsics["W"])
    fx, fy = intrinsics["fx"], intrinsics["fy"]
    cx, cy = intrinsics["cx"], intrinsics["cy"]
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)

    model = GaussianModel(0)
    opt_args = OptimizationParams(ArgumentParser())
    model.training_setup(opt_args)
    model.load_ply(gaussian_model_path)

    available_frames = set(p.stem.split('_')[0] for p in (dataset_path / "rgb").glob("*.jpg"))

    for i, kf_id in enumerate(tqdm(kf_ids, desc="Rendering")):
        kf_str = f"{kf_id:06d}"
        rgb_path = dataset_path / "rgb" / f"{kf_str}_rgb.jpg"
        if not rgb_path.exists():
            continue

        gt_img = imageio.imread(rgb_path)
        c2w = c2ws[i].numpy()
        w2c = np.linalg.inv(c2w)

        render_out = render_gaussian_model(model, get_render_settings(W, H, K, w2c))
        rendered = render_out["color"].squeeze().permute(1, 2, 0).detach().cpu().numpy()
        rendered = np.clip(rendered * 255, 0, 255).astype(np.uint8)

        combined = np.concatenate([gt_img, rendered], axis=1)
        imageio.imwrite(out_dir / f"compare_{kf_str}.jpg", combined)

if __name__ == "__main__":
    ckpt_dir = Path("output/ReplicaMultiagent/office0")
    dataset_path = Path("/home/jliu/MAGiC_SLAM/Data/Office-0/office_0_part1")
    out_dir = ckpt_dir / "figures"
    gaussian_model_path = ckpt_dir / "merged_refined.ply"
    config = load_config(ckpt_dir / "config.yaml")
    intrinsics = config["cam"]

    poses = load_camera_poses(ckpt_dir)
    render_and_save_images(gaussian_model_path, poses, dataset_path, intrinsics, out_dir)
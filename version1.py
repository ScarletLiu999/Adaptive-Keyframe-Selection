# keyframe_selector.py
from transformers import AutoImageProcessor, AutoModel
import numpy as np
import yaml
from collections import deque
import open3d as o3d
import torch
import glob
import os
import wandb
from PIL import Image
from src.entities.loop_detection.netvlad import NetVLAD  

class KeyframeSelector:
    def __init__(self, N=15, m=10):
        self.N = N
        self.m = m
        self.window = deque(maxlen=N)
        self.keyframe_set = []
        self.rgb_paths = []
        self.phi_k_set = np.empty((0, 256))
        self.delta_history = [0.5]
        self.prev_delta = 0
        self.t = 0
        conf = {
            'model_name': 'VGG16-NetVLAD-Pitts30K',
            'whiten': True,
            'checkpoint_path': os.path.expanduser('~/weights/Pitts30K_struct.mat')
        }
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.feature_extractor = NetVLAD(conf).to(device)
        self.feature_extractor.eval()

    def compute_descriptor(self, rgb_path):
        img = Image.open(rgb_path).convert('RGB')
        img_array = np.array(img)
        if img_array.shape[-1] != 3:
            raise ValueError(f"Image {rgb_path} is not RGB: shape={img_array.shape}")
        img = torch.tensor(img_array, dtype=torch.float32)
        img = self.feature_extractor.transform(img).to(self.feature_extractor.backbone[0].weight.device)
        with torch.no_grad():
            desc = self.feature_extractor(img)
        return desc.squeeze().cpu().numpy()


    def compute_marginal_gain(self, phi_et):
        if len(self.keyframe_set) == 0:
            return float('inf')
        inner_products = np.dot(self.phi_k_set, phi_et)
        m = min(len(self.phi_k_set), self.m)
        top_m_indices = np.argsort(-inner_products)[:m]
        distances = np.sqrt(2 - 2 * inner_products[top_m_indices])
        return np.min(distances)

    def compute_feature_change_rate(self, phi_et, phi_et_minus_1):
        dot_product = np.dot(phi_et, phi_et_minus_1)
        dot_product = np.clip(dot_product, -1.0, 1.0)
        distance = np.sqrt(2 - 2 * dot_product)

        # distance = np.sqrt(2 - 2 * np.dot(phi_et, phi_et_minus_1))
        if self.t == 1:
            return distance
        return 0.8 * self.prev_delta + 0.2 * distance

    def normalize_feature_change_rate(self, delta_t):
        max_delta = max(self.delta_history)
        min_delta = min(self.delta_history)
        if max_delta == min_delta:
            return 0.5
        return (delta_t - min_delta) / (max_delta - min_delta)

    def compute_dynamic_threshold(self, normalized_delta):
        alpha_t = 0.1 * (1 + 0.7 * normalized_delta)
        return np.clip(alpha_t, 0.05, 0.2)

    def select_keyframe(self, rgb_path, point_cloud):
        self.t += 1
        phi_et = self.compute_descriptor(rgb_path)
        self.window.append(point_cloud)
        self.rgb_paths.append(rgb_path)

        if self.t > 1:
            phi_et_minus_1 = self.compute_descriptor(self.rgb_paths[-2])
            delta_t = self.compute_feature_change_rate(phi_et, phi_et_minus_1)
        else:
            delta_t = self.compute_feature_change_rate(phi_et, phi_et)
        self.delta_history.append(delta_t)
        self.prev_delta = delta_t

        normalized_delta = self.normalize_feature_change_rate(delta_t)
        alpha_t = self.compute_dynamic_threshold(normalized_delta)
        marginal_gain = self.compute_marginal_gain(phi_et)

        if marginal_gain >= alpha_t:
            self.keyframe_set.append(point_cloud)
            self.phi_k_set = np.vstack([self.phi_k_set, phi_et])
            return True, marginal_gain, delta_t, alpha_t
        return False, marginal_gain, delta_t, alpha_t

def rgbd_to_point_cloud(rgb_path, depth_path, intrinsics, depth_scale=6553.5):
    rgb = o3d.io.read_image(rgb_path)
    depth = o3d.io.read_image(depth_path)
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        rgb, depth, depth_scale=depth_scale, convert_rgb_to_intensity=False
    )
    intrinsic = o3d.camera.PinholeCameraIntrinsic(
        intrinsics["width"], intrinsics["height"], intrinsics["fx"], intrinsics["fy"], intrinsics["cx"], intrinsics["cy"]
    )
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsic)
    return pcd

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    if 'inherit_from' in config:
        base_path = os.path.dirname(config_path)
        parent_config_path = os.path.join(base_path, config['inherit_from'])
        parent_config = load_config(parent_config_path)
        for key, value in config.items():
            if isinstance(value, dict) and key in parent_config:
                parent_config[key].update(value)
            else:
                parent_config[key] = value
        config = parent_config
    return config

def main():
    config_path = os.path.expanduser("~/MAGiC-SLAM/configs/ReplicaMultiagent/office_0.yaml")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file {config_path} not found!")
    config = load_config(config_path)
    intrinsics = {
        "width": config["camera"]["W"],
        "height": config["camera"]["H"],
        "fx": config["camera"]["fx"],
        "fy": config["camera"]["fy"],
        "cx": config["camera"]["cx"],
        "cy": config["camera"]["cy"]
    }
    depth_scale = config["camera"]["depth_scale"]
    frame_limit = config["data"]["frame_limit"]
    print(f"Loaded config: intrinsics={intrinsics}, depth_scale={depth_scale}, frame_limit={frame_limit}")

    wandb.init(project="keyframe_test", config={"N": 15, "m": 5, "alpha_min": 0.01, "alpha_max": 0.05})
    selector = KeyframeSelector(
        N=15,
        m=5,
        feature_extractor_name=config["loop_detection"]["feature_extractor_name"],
        weights_path=config["loop_detection"]["weights_path"],
        embed_size=config["loop_detection"]["embed_size"]
    )

    base_path = os.path.expanduser("Data/Office-0/office_0_part1/results")
    rgb_files = sorted(glob.glob(f"{base_path}/frame*.jpg"))[:frame_limit]
    depth_files = sorted(glob.glob(f"{base_path}/depth*.png"))[:frame_limit]
    if len(rgb_files) != len(depth_files):
        print(f"Error: RGB ({len(rgb_files)}) and Depth ({len(depth_files)}) file counts do not match!")
        return

    for rgb_path, depth_path in zip(rgb_files, depth_files):
        point_cloud = rgbd_to_point_cloud(rgb_path, depth_path, intrinsics)
        is_keyframe, marginal_gain, delta_t, alpha_t = selector.select_keyframe(rgb_path, point_cloud)
        wandb.log({
            "keyframe_count": len(selector.keyframe_set),
            "marginal_gain": marginal_gain,
            "delta_t": delta_t,
            "alpha_t": alpha_t
        })

    for i, kf in enumerate(selector.keyframe_set):
        o3d.io.write_point_cloud(f"keyframe_{i}.ply", kf)
    print(f"Saved {len(selector.keyframe_set)} keyframes.")

if __name__ == "__main__":
    main()
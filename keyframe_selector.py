# keyframe_selector.py (clean version)
import os
import glob
import yaml
import torch
import wandb
import numpy as np
import open3d as o3d
from PIL import Image
from collections import deque
from transformers import AutoImageProcessor, AutoModel


class KeyframeSelector:
    def __init__(self, config):
        loop_cfg = config["loop_detection"]

        self.m = loop_cfg.get("m", 5)
        self.embed_size = loop_cfg.get("embed_size", 384)
        self.device = loop_cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")

        self.keyframe_set = []
        self.rgb_paths = []
        self.phi_k_set = np.empty((0, self.embed_size))
        self.delta_history = [0.5]
        self.prev_delta = 0
        self.t = 0

        feature_extractor_name = loop_cfg.get("feature_extractor_name", "dino")
        weights_path = loop_cfg.get("weights_path", "facebook/dinov2-small")

        if feature_extractor_name == "dino":
            self.processor = AutoImageProcessor.from_pretrained(weights_path, use_fast=True)
            self.feature_extractor = AutoModel.from_pretrained(weights_path).to(self.device)
            self.transform = lambda img: self.processor(img, return_tensors="pt")["pixel_values"].to(self.device)
        else:
            raise ValueError(f"Unsupported feature extractor: {feature_extractor_name}")

        self.feature_extractor.eval()

    def compute_descriptor(self, rgb_path):
        img = Image.open(rgb_path).convert("RGB")
        img_tensor = self.transform(img)
        with torch.no_grad():
            output = self.feature_extractor(img_tensor)
            features = output.last_hidden_state[:, 1:, :] if hasattr(output, "last_hidden_state") else output[0][:, 1:, :]
            embedding = features.mean(dim=1)
            embedding = torch.nn.functional.normalize(embedding, dim=1)
        return embedding.squeeze().cpu().numpy()

    def compute_marginal_gain(self, phi_et):
        if len(self.keyframe_set) == 0:
            return float("inf")
        inner_products = np.dot(self.phi_k_set, phi_et)
        top_m = np.argsort(-inner_products)[:min(len(inner_products), self.m)]
        similarities = np.clip(inner_products[top_m], -1.0, 1.0)
        distances = np.sqrt(2 - 2 * similarities)
        return np.min(distances)

    def compute_feature_change_rate(self, phi_et, phi_prev):
        sim = np.clip(np.dot(phi_et, phi_prev), -1.0, 1.0)
        dist = np.sqrt(2 - 2 * sim)
        return dist if self.t == 1 else 0.8 * self.prev_delta + 0.2 * dist

    def normalize_feature_change_rate(self, delta_t):
        return 0.5 if max(self.delta_history) == min(self.delta_history) else (delta_t - min(self.delta_history)) / (max(self.delta_history) - min(self.delta_history))
    
    
    def compute_dynamic_threshold(self, normalized_delta):
        alpha_t = 0.1 * (1 + 0.7 * normalized_delta)
        return np.clip(alpha_t, 0.05, 0.2)

        #alpha_t = 0.01 * (1 + 0.7 * normalized_delta)  # 进一步降低基线
        #return np.clip(alpha_t, 0.005, 0.025)

        #alpha_t = 0.0005 * (1 + 0.1 * normalized_delta)
        #return np.clip(alpha_t, 0.0002, 0.001)

        #alpha_t = 0.0001 * (1 + 0.5 * normalized_delta)
        #return np.clip(alpha_t, 0.00005, 0.0002)

    def select_keyframe(self, rgb_path, point_cloud):
        self.t += 1
        phi_et = self.compute_descriptor(rgb_path)
        self.rgb_paths.append(rgb_path)

        phi_prev = self.compute_descriptor(self.rgb_paths[-2]) if self.t > 1 else phi_et
        delta_t = self.compute_feature_change_rate(phi_et, phi_prev)
        self.delta_history.append(delta_t)
        self.prev_delta = delta_t

        normalized_delta = self.normalize_feature_change_rate(delta_t)
        alpha_t = self.compute_dynamic_threshold(normalized_delta)
        marginal_gain = self.compute_marginal_gain(phi_et)

        #if marginal_gain >= alpha_t:
        if marginal_gain >= 0.08:
            self.keyframe_set.append(point_cloud)
            self.phi_k_set = np.vstack([self.phi_k_set, phi_et])
            return True, marginal_gain, delta_t, alpha_t
        return False, marginal_gain, delta_t, alpha_t


def rgbd_to_point_cloud(rgb_path, depth_path, intrinsics, depth_scale):
    rgb = o3d.io.read_image(rgb_path)
    depth = o3d.io.read_image(depth_path)
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(rgb, depth, depth_scale=depth_scale, convert_rgb_to_intensity=False)
    intrinsic = o3d.camera.PinholeCameraIntrinsic(
        intrinsics["W"], intrinsics["H"], intrinsics["fx"], intrinsics["fy"], intrinsics["cx"], intrinsics["cy"]
    )
    return o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsic)


def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    if 'inherit_from' in config:
        parent_config = load_config(os.path.join(os.path.dirname(config_path), config['inherit_from']))
        parent_config.update(config)
        return parent_config
    return config


def main():
    config_path = os.path.expanduser("/home/jliu/MAGiC-SLAM/configs/ReplicaMultiagent/office_0.yaml")
    config = load_config(config_path)

    intrinsics = config["camera"]
    depth_scale = config["camera"]["depth_scale"]
    frame_limit = config["data"].get("frame_limit", 1000)

    base_path = os.path.expanduser("Data/Office-0/office_0_part1/results")
    rgb_files = sorted(glob.glob(f"{base_path}/frame*.jpg"))[:frame_limit]
    depth_files = sorted(glob.glob(f"{base_path}/depth*.png"))[:frame_limit]

    agent_stride = 50
    num_agents = len(rgb_files) // agent_stride
    total_keyframes = 0

    for agent_id in range(num_agents):
        print(f"\n[Agent {agent_id}] Processing frames {agent_id * agent_stride} – {(agent_id + 1) * agent_stride - 1}")

        selector = KeyframeSelector(config)
        marginal_gains = []

        for i in range(agent_id * agent_stride, (agent_id + 1) * agent_stride):
            rgb_path = rgb_files[i]
            depth_path = depth_files[i]
            point_cloud = rgbd_to_point_cloud(rgb_path, depth_path, intrinsics, depth_scale)

            is_keyframe, marginal_gain, delta_t, alpha_t = selector.select_keyframe(rgb_path, point_cloud)
            marginal_gains.append(marginal_gain)

        
        output_dir = f"keyframes/agent_{agent_id}"
        os.makedirs(output_dir, exist_ok=True)

        for j, kf in enumerate(selector.keyframe_set):
            filename = os.path.join(output_dir, f"keyframe_{j}.ply")
            o3d.io.write_point_cloud(filename, kf)


        print(f"[Agent {agent_id}] Saved {len(selector.keyframe_set)} keyframes.")
        total_keyframes += len(selector.keyframe_set)

        valid_gains = [g for g in marginal_gains if not np.isinf(g)]
        if valid_gains:
            print(f"[Agent {agent_id}] marginal_gain: max={max(valid_gains):.6f}, min={min(valid_gains):.6f}, mean={np.mean(valid_gains):.6f}")
        else:
            print(f"[Agent {agent_id}] No valid marginal_gain values.")

    print(f"\nTotal keyframes across all {num_agents} agents: {total_keyframes}")


if __name__ == "__main__":
    main()

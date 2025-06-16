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
    def __init__(self, m=5, feature_extractor_name="dino", weights_path="facebook/dinov2-base", embed_size=768):
        #self.N = N
        self.m = m
        #self.window = deque(maxlen=N)
        self.keyframe_set = []
        self.rgb_paths = []
        self.phi_k_set = np.empty((0, embed_size))
        self.delta_history = [0.5]
        self.prev_delta = 0
        self.t = 0
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if feature_extractor_name == "dino":
            self.processor = AutoImageProcessor.from_pretrained(weights_path, use_fast=True)
            self.feature_extractor = AutoModel.from_pretrained(weights_path).to(device)
            self.transform = lambda img: self.processor(img, return_tensors="pt")['pixel_values'].to(device)
        else:
            raise ValueError(f"Unsupported feature extractor: {feature_extractor_name}")
        self.feature_extractor.eval()
        self.embed_size = embed_size


    def compute_descriptor(self, rgb_path):
        img = Image.open(rgb_path).convert('RGB')
        img_array = np.array(img)
        if img_array.shape[-1] != 3:
            raise ValueError(f"Image {rgb_path} is not RGB: shape={img_array.shape}")
        img_tensor = self.transform(img)  # 用定义好的 transform 预处理
        with torch.no_grad():
            output = self.feature_extractor(img_tensor)
            if hasattr(output, "last_hidden_state"):
                features = output.last_hidden_state[:, 1:, :]  # 去掉 CLS token
            else:
                features = output[0][:, 1:, :]
            embedding = features.mean(dim=1)  # Patch token 平均
            embedding = torch.nn.functional.normalize(embedding, dim=1)  # L2 归一化
        return embedding.squeeze().cpu().numpy()


    def compute_marginal_gain(self, phi_et):
        if len(self.keyframe_set) == 0:
            return float('inf')
        inner_products = np.dot(self.phi_k_set, phi_et)
        m = min(len(self.phi_k_set), self.m)
        top_m_indices = np.argsort(-inner_products)[:m]
        
        dot_products = inner_products[top_m_indices]
        dot_products = np.clip(dot_products, -1.0, 1.0)  # ensure valid range
        distances = np.sqrt(2 - 2 * dot_products)
                
        #distances = np.sqrt(2 - 2 * inner_products[top_m_indices])
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

        #alpha_t = 0.01 * (1 + 0.7 * normalized_delta)  # 进一步降低基线
        #return np.clip(alpha_t, 0.005, 0.025)

        #alpha_t = 0.0005 * (1 + 0.1 * normalized_delta)
        #return np.clip(alpha_t, 0.0002, 0.001)

        #alpha_t = 0.0001 * (1 + 0.5 * normalized_delta)
        #return np.clip(alpha_t, 0.00005, 0.0002)
    

    def select_keyframe(self, rgb_path, point_cloud):
        self.t += 1
        phi_et = self.compute_descriptor(rgb_path)
        #print(f"[Descriptor] Frame {self.t}: {phi_et[:5]}")

        #self.window.append(point_cloud)
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
        #if marginal_gain >= 0.08:
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
    config_path = os.path.abspath(config_path)
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    if 'inherit_from' in config:
        inherit_path = config['inherit_from']
        # If the path is not absolute, resolve it relative to the current config
        if not os.path.isabs(inherit_path):
            inherit_path = os.path.join(os.path.dirname(config_path), inherit_path)
        parent_config = load_config(os.path.abspath(inherit_path))
        for key, value in config.items():
            if isinstance(value, dict) and key in parent_config:
                parent_config[key].update(value)
            else:
                parent_config[key] = value
        config = parent_config

    return config

def main():

    config_path = os.path.expanduser("/home/jliu/MAGiC-SLAM/configs/ReplicaMultiagent/office_0.yaml")
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

    wandb.init(project="keyframe_test", config={"agent_stride": 50, "alpha_min": 0.0002, "alpha_max": 0.001})
    print(f"Loaded config: intrinsics={intrinsics}, depth_scale={depth_scale}, frame_limit={frame_limit}")

    base_path = os.path.expanduser("Data/Office-0/office_0_part1/results")
    rgb_files = sorted(glob.glob(f"{base_path}/frame*.jpg"))[:frame_limit]
    depth_files = sorted(glob.glob(f"{base_path}/depth*.png"))[:frame_limit]

    if len(rgb_files) != len(depth_files):
        raise ValueError(f"RGB ({len(rgb_files)}) and Depth ({len(depth_files)}) counts do not match.")

    agent_stride = 50
    num_agents = len(rgb_files) // agent_stride
    total_keyframes = 0

    for agent_id in range(num_agents):
        print(f"\n[Agent {agent_id}] Processing frames {agent_id * agent_stride} – {(agent_id + 1) * agent_stride - 1}")

        # 每个 agent 拥有独立的 KeyframeSelector（也可以移出循环保持全局选择器）
        selector = KeyframeSelector(
            m=5,
            feature_extractor_name=config["loop_detection"]["feature_extractor_name"],
            weights_path=config["loop_detection"]["weights_path"],
            embed_size=config["loop_detection"]["embed_size"]
        )

        marginal_gains = []

        for i in range(agent_id * agent_stride, (agent_id + 1) * agent_stride):
            rgb_path = rgb_files[i]
            depth_path = depth_files[i]
            point_cloud = rgbd_to_point_cloud(rgb_path, depth_path, intrinsics)

            is_keyframe, marginal_gain, delta_t, alpha_t = selector.select_keyframe(rgb_path, point_cloud)
            marginal_gains.append(marginal_gain)

            wandb.log({
                "agent_id": agent_id,
                "frame_index": i,
                "marginal_gain": marginal_gain,
                "alpha_t": alpha_t,
                "delta_t": delta_t,
                "keyframe_count": len(selector.keyframe_set)
            })

        # 保存关键帧
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
import os
import re
from pathlib import Path

def extract_keyframes_by_agent(raw_txt_path, output_dir):
    with open(raw_txt_path, 'r') as f:
        lines = f.readlines()

    os.makedirs(output_dir, exist_ok=True)

    current_agent = None
    agent_frames = []

    def save_agent(agent_id, frames):
        indices = []
        for name in frames:
            match = re.search(r'(\d+)', name)
            if match:
                indices.append(int(match.group(1)))
        indices.sort()
        out_path = Path(output_dir) / f"agent{agent_id}.txt"
        with open(out_path, 'w') as f:
            for idx in indices:
                f.write(f"{idx}\n")

    for line in lines:
        line = line.strip()
        if not line:
            continue
        if line.startswith("Agent"):
            if current_agent is not None:
                save_agent(current_agent, agent_frames)
            current_agent = int(re.search(r'\d+', line).group(0))
            agent_frames = []
        else:
            agent_frames.append(line)

    if current_agent is not None:
        save_agent(current_agent, agent_frames)

    print(f"[✔] Done. Saved agent keyframes to {output_dir}")

# 用法（你可以直接运行）
extract_keyframes_by_agent("keyframes.txt", "configs/keyframes/office_0_part1")

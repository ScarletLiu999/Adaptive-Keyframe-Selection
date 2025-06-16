import numpy as np
from PIL import Image
from transformers import AutoImageProcessor, AutoModel
import torch
import glob
import os

device = 'cuda' if torch.cuda.is_available() else 'cpu'
processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base", use_fast=True)
model = AutoModel.from_pretrained("facebook/dinov2-base").to(device)
model.eval()

base_path = os.path.expanduser("Data/Office-0/office_0_part1/results")
rgb_files = sorted(glob.glob(f"{base_path}/frame*.jpg"))[:20] 

descriptors = []
for rgb_path in rgb_files:
    img = Image.open(rgb_path).convert('RGB')
    inputs = processor(img.resize((224, 224)), return_tensors="pt")['pixel_values'].to(device)
    with torch.no_grad():
        outputs = model(inputs)
        desc = outputs.last_hidden_state[:, 1:, :].mean(dim=1)  # 忽略 [CLS]
        desc = torch.nn.functional.normalize(desc, dim=1)
    descriptors.append(desc.squeeze().cpu().numpy())

for i in range(len(descriptors)-1):
    dist = np.sqrt(2 - 2 * np.dot(descriptors[i], descriptors[i+1]))
    print(f"Descriptor distance between frame {i} and {i+1}: {dist:.4f}")
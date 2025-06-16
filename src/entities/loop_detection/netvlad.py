""" We borrowed this feature extractor from CP-SLAM to compare it with ours. """
from pathlib import Path
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from scipy.io import loadmat


logger = logging.getLogger(__name__)

EPS = 1e-6


class NetVLADLayer(nn.Module):
    def __init__(self, input_dim=512, K=64, score_bias=False, intranorm=True):
        super().__init__()
        self.score_proj = nn.Conv1d(input_dim, K, kernel_size=1, bias=score_bias)
        centers = nn.parameter.Parameter(torch.empty([input_dim, K]))
        nn.init.xavier_uniform_(centers)
        self.register_parameter('centers', centers)
        self.intranorm = intranorm
        self.output_dim = input_dim * K

    def forward(self, x):
        b = x.size(0)
        scores = self.score_proj(x)
        scores = F.softmax(scores, dim=1)
        diff = (x.unsqueeze(2) - self.centers.unsqueeze(0).unsqueeze(-1))
        desc = (scores.unsqueeze(1) * diff).sum(dim=-1)
        if self.intranorm:
            desc = F.normalize(desc, dim=1)
        desc = desc.view(b, -1)
        desc = F.normalize(desc, dim=1)
        return desc
    

class NetVLAD(nn.Module):\

    default_conf = {'model_name': 'VGG16-NetVLAD-Pitts30K', 'whiten': True}
    required_inputs = ['image']
    dir_models = {
        'VGG16-NetVLAD-Pitts30K': 'https://cvg-data.inf.ethz.ch/hloc/netvlad/Pitts30K_struct.mat',
        'VGG16-NetVLAD-TokyoTM': 'https://cvg-data.inf.ethz.ch/hloc/netvlad/TokyoTM_struct.mat'
    }

    def __init__(self, conf):
        super().__init__()
        checkpoint = Path(conf['checkpoint_path'])
        backbone = list(models.vgg16().children())[0]
        self.backbone = nn.Sequential(*list(backbone.children())[: -2])
        self.netvlad = NetVLADLayer()
        if conf['whiten']:
            self.whiten = nn.Linear(self.netvlad.output_dim, 4096)
        self.desc_dim_reducer = nn.Linear(4096 if conf['whiten'] else self.netvlad.output_dim, 256)  # 降维到 256

        mat = loadmat(str(checkpoint), struct_as_record=False, squeeze_me=True)
        for layer, mat_layer in zip(self.backbone.children(), mat['net'].layers):
            if isinstance(layer, nn.Conv2d):
                w = mat_layer.weights[0]
                b = mat_layer.weights[1]
                w = torch.tensor(w).float().permute([3, 2, 0, 1])
                b = torch.tensor(b).float()
                layer.weight = nn.Parameter(w)
                layer.bias = nn.Parameter(b)

        score_w = mat['net'].layers[30].weights[0]
        center_w = -mat['net'].layers[30].weights[1]
        score_w = torch.tensor(score_w).float().permute([1, 0]).unsqueeze(-1)
        center_w = torch.tensor(center_w).float()
        self.netvlad.score_proj.weight = nn.Parameter(score_w)
        self.netvlad.centers = nn.Parameter(center_w)

        if conf['whiten']:
            w = mat['net'].layers[33].weights[0]
            b = mat['net'].layers[33].weights[1]
            w = torch.tensor(w).float().squeeze().permute([1, 0])
            b = torch.tensor(b.squeeze()).float()
            self.whiten.weight = nn.Parameter(w)
            self.whiten.bias = nn.Parameter(b)

        self.preprocess = {
            'mean': mat['net'].meta.normalization.averageImage[0, 0],
            'std': np.array([1, 1, 1], dtype=np.float32)
        }
        
    def transform(self, img: torch.Tensor):
        img = (img / 255.)
        img = torch.permute(img, (2, 0, 1)).unsqueeze(0)
        return torch.clamp(img, min=0, max=1)

    def forward(self, image: torch.Tensor):
        assert image.shape[1] == 3  # 输入应为 (batch_size, 3, height, width)
        mean = self.preprocess['mean']
        std = self.preprocess['std']
        image = image - image.new_tensor(mean).view(1, -1, 1, 1)
        image = image / image.new_tensor(std).view(1, -1, 1, 1)

        descriptors = self.backbone(image)
        b, c, _, _ = descriptors.size()
        descriptors = descriptors.view(b, c, -1)
        descriptors = F.normalize(descriptors, dim=1)
        desc = self.netvlad(descriptors)
        if hasattr(self, 'whiten'):
            desc = self.whiten(desc)
            desc = F.normalize(desc, dim=1)
        desc = self.desc_dim_reducer(desc)
        desc = F.normalize(desc, dim=1)
        return desc

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from torchvision import transforms
from torchvision.transforms import functional as TF

class MaskTransform:
    def __init__(self, size=(224, 224)):
        self.size = size

    def __call__(self, mask):
        # Resize using NEAREST interpolation to preserve class values
        mask = TF.resize(mask, self.size, interpolation=transforms.InterpolationMode.NEAREST)
        # Convert to LongTensor of class indices
        return torch.as_tensor(np.array(mask), dtype=torch.long)

mask_transform = MaskTransform(size=(224, 224))

img_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from torchvision import transforms
from torchvision.transforms import functional as TF

#------------------------- MODEL 1 ( BUILDING DETECTION ) ----------------------
# Reinitialize encoder
def get_resnet34_encoder():
    resnet34 = models.resnet34(pretrained=True)
    resnet34.conv1 = nn.Conv2d(in_channels=6, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False
)
    return resnet34
def get_resnet34_destruction_encoder():
    resnet34 = models.resnet34(pretrained=True)
    resnet34.conv1 = nn.Conv2d(in_channels=7, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False
)
    return resnet34

# Decoder conv block
def conv_block(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.BatchNorm2d(out_channels),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.BatchNorm2d(out_channels)
    )

# Correctly aligned UNet model
class UNetModel(nn.Module):
    def __init__(self, n_classes=1):
        super().__init__()

        self.encoder = get_resnet34_encoder()

        self.up5 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec5 = conv_block(256 + 256, 256)

        self.up4 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec4 = conv_block(128 + 128, 128)

        self.up3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec3 = conv_block(64 + 64, 64)

        self.up2 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec2 = conv_block(32 + 64, 64)

        self.up1 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        self.final_conv = nn.Conv2d(64, n_classes, kernel_size=1)

        self.dropout = nn.Dropout2d(0.2)

    def forward(self, x):
        # Encoder
        e1 = self.encoder.relu(self.encoder.bn1(self.encoder.conv1(x)))  # 64
        e2 = self.encoder.layer1(self.encoder.maxpool(e1))  # 64
        e3 = self.encoder.layer2(e2)  # 128
        e4 = self.encoder.layer3(e3)  # 256
        e5 = self.encoder.layer4(e4)  # 512

        # Decoder
        d5 = self.dropout(self.dec5(torch.cat([self.up5(e5), e4], dim=1)))
        d4 = self.dropout(self.dec4(torch.cat([self.up4(d5), e3], dim=1)))
        d3 = self.dropout(self.dec3(torch.cat([self.up3(d4), e2], dim=1)))
        d2 = self.dropout(self.dec2(torch.cat([self.up2(d3), e1], dim=1)))
        d1 = self.up1(d2)

        out = self.final_conv(d1)
        return F.interpolate(out, size=x.shape[2:], mode='bilinear', align_corners=False)
class UNetModelDestruction(nn.Module):
    def __init__(self, n_classes=1):
        super().__init__()

        self.encoder = get_resnet34_destruction_encoder()
        self.up5 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec5 = conv_block(256 + 256, 256)

        self.up4 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec4 = conv_block(128 + 128, 128)

        self.up3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec3 = conv_block(64 + 64, 64)
        

        self.up2 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec2 = conv_block(32 + 64, 64)

        self.up1 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        self.final_conv = nn.Conv2d(64, n_classes, kernel_size=1)

        self.dropout = nn.Dropout2d(0.4)

    def forward(self, x):
        assert x.shape[1] == 7, f"Expected 7 channels, got {x.shape[1]}"
        # Encoder
        e1 = self.encoder.relu(self.encoder.bn1(self.encoder.conv1(x)))  # 64
        e2 = self.encoder.layer1(self.encoder.maxpool(e1))  # 64
        e3 = self.encoder.layer2(e2)  # 128
        e4 = self.encoder.layer3(e3)  # 256
        e5 = self.encoder.layer4(e4)  # 512

        # Decoder
        d5 = self.dropout(self.dec5(torch.cat([self.up5(e5), e4], dim=1)))
        d4 = self.dropout(self.dec4(torch.cat([self.up4(d5), e3], dim=1)))
        d3 = self.dropout(self.dec3(torch.cat([self.up3(d4), e2], dim=1)))
        d2 = self.dropout(self.dec2(torch.cat([self.up2(d3), e1], dim=1)))
        d1 = self.up1(d2)

        out = self.final_conv(d1)
        return F.interpolate(out, size=x.shape[2:], mode='bilinear', align_corners=False)

# Function to return an instance of the model
def get_model(n_classes=1):
    return UNetModel(n_classes)

def get_model_destruction(n_classes=1):
    return UNetModel(n_classes)

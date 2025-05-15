import torch
import torch.nn as nn
from torchvision.models import convnext_base, ConvNeXt_Base_Weights

class SiameseNetwork(nn.Module):
    def __init__(self, embedding_dim=128):
        super(SiameseNetwork, self).__init__()
        self.backbone = convnext_base(weights=ConvNeXt_Base_Weights.IMAGENET1K_V1)
        self.backbone.classifier = nn.Identity()
        self.pooling = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(1024, embedding_dim)

    def forward_one(self, x):
        x = self.backbone(x)
        x = self.pooling(x).flatten(1)
        x = self.fc(x)
        return x

    def forward(self, x1, x2):
        output1 = self.forward_one(x1)
        output2 = self.forward_one(x2)
        return output1, output2
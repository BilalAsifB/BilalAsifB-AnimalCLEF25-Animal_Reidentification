import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import convnext_base, ConvNeXt_Base_Weights

class ProtoNet(nn.Module):
    def __init__(self, embedding_size=1024, num_classes=3):
        super(ProtoNet, self).__init__()
        self.backbone = convnext_base(weights=ConvNeXt_Base_Weights.IMAGENET1K_V1)
        self.backbone.classifier = nn.Identity()
        self.pooling = nn.AdaptiveAvgPool2d((1, 1))
        self.embedding = nn.Linear(1024, embedding_size)

    def forward(self, x, return_embeddings=False):
        x = self.backbone(x)
        x = self.pooling(x).flatten(1)
        embeddings = self.embedding(x)
        embeddings = F.normalize(embeddings, p=2, dim=1)
        if return_embeddings:
            return embeddings
        return embeddings
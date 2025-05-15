import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import convnext_base, ConvNeXt_Base_Weights

class GeM(nn.Module):
    def __init__(self, p=3.0, eps=1e-6):
        super(GeM, self).__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return F.avg_pool2d(x.clamp(min=self.eps).pow(self.p), (x.size(-2), x.size(-1))).pow(1./self.p)

class ArcMarginProduct(nn.Module):
    def __init__(self, in_features, out_features, s=30.0, m=0.50):
        super(ArcMarginProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, input, label):
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        theta = torch.acos(torch.clamp(cosine, -1.0 + 1e-7, 1.0 - 1e-7))
        target_logit = torch.cos(theta + self.m)
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, label.view(-1, 1), 1)
        output = cosine * (1 - one_hot) + target_logit * one_hot
        output *= self.s
        return output

class AnimalModel(nn.Module):
    def __init__(self, embedding_size=512, num_classes=3, arc_s=30.0, arc_m=0.50):
        super(AnimalModel, self).__init__()
        self.backbone = convnext_base(weights=ConvNeXt_Base_Weights.IMAGENET1K_V1)
        self.backbone.classifier = nn.Identity()
        self.pooling = GeM()
        self.embedding = nn.Linear(1024, embedding_size)
        self.metric = ArcMarginProduct(embedding_size, num_classes, s=arc_s, m=arc_m)

    def forward(self, x, label=None):
        x = self.backbone(x)
        x = self.pooling(x).flatten(1)
        x = self.embedding(x)
        if label is not None:
            x = self.metric(x, label)
        return x
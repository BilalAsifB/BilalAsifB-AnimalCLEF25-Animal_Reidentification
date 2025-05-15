import torch

def contrastive_loss(output1, output2, label, margin=1.0):
    distance = torch.nn.functional.pairwise_distance(output1, output2)
    loss_same = label * torch.pow(distance, 2)
    loss_diff = (1 - label) * torch.pow(torch.clamp(margin - distance, min=0.0), 2)
    return (loss_same + loss_diff).mean()
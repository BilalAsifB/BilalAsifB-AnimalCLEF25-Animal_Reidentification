import torch
from torch.cuda.amp import autocast, GradScaler
import gc

def train_one_epoch(model, loader, optimizer, device, scaler, config):
    model.train()
    total_loss = 0
    optimizer.zero_grad()
    for i, (imgs, labels, _) in enumerate(loader):
        imgs, labels = imgs.to(device), labels.to(device)
        with autocast():
            outputs = model(imgs, labels)
            loss = torch.nn.functional.cross_entropy(outputs, labels) / config['accum_steps']
        scaler.scale(loss).backward()
        if (i + 1) % config['accum_steps'] == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        total_loss += loss.item() * config['accum_steps']
    torch.cuda.empty_cache()
    gc.collect()
    return total_loss / len(loader)
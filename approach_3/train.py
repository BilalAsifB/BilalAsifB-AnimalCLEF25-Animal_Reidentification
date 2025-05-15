import torch
from torch.cuda.amp import autocast, GradScaler
import gc

def pretrain_one_epoch(model, loader, optimizer, scaler, device):
    model.train()
    total_loss = 0
    total_correct = 0
    total_samples = 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        with autocast():
            outputs = model(imgs)
            loss = torch.nn.functional.cross_entropy(outputs, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item()
        preds = outputs.argmax(dim=1)
        total_correct += (preds == labels).sum().item()
        total_samples += labels.size(0)
    return total_loss / len(loader), total_correct / total_samples

def train_one_episode(model, support_imgs, support_labels, query_imgs, query_labels, optimizer, scaler, device, config):
    model.train()
    batch_size = support_imgs.size(0)
    support_max_len = support_imgs.size(1)
    query_max_len = query_imgs.size(1)
    support_imgs = support_imgs.view(batch_size * support_max_len, 3, config['image_size'][0], config['image_size'][1])
    query_imgs = query_imgs.view(batch_size * query_max_len, 3, config['image_size'][0], config['image_size'][1])
    
    with autocast():
        support_embs = model(support_imgs, return_embeddings=True)
        query_embs = model(query_imgs, return_embeddings=True)
    
    support_embs = support_embs.view(batch_size, support_max_len, config['embedding_size'])
    query_embs = query_embs.view(batch_size, query_max_len, config['embedding_size'])
    
    support_labels_flat = support_labels.view(-1)
    query_labels_flat = query_labels.view(-1)
    
    optimizer.zero_grad()
    with autocast():
        from .loss import prototypical_loss, contrastive_loss
        loss_proto, acc = prototypical_loss(support_embs, support_labels_flat, query_embs, query_labels_flat, config['n_way_train'], config['k_shot'])
        loss_contrast = contrastive_loss(support_embs.view(-1, config['embedding_size']), support_labels_flat)
        loss = loss_proto + 0.5 * loss_contrast
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    return loss.item(), acc
import torch
import torch.nn.functional as F

def compute_prototypes(embeddings, labels, n_way, k_shot):
    prototypes = torch.zeros(n_way, embeddings.size(-1)).to(embeddings.device)
    for i in range(n_way):
        mask = labels == i
        if mask.any():
            prototypes[i] = embeddings[mask].mean(dim=0)
    return prototypes

def contrastive_loss(support_embs, support_labels):
    margin = 1.0
    distances = torch.cdist(support_embs, support_embs)
    same_label = (support_labels.unsqueeze(1) == support_labels.unsqueeze(0)).float()
    loss = same_label * distances**2 + (1 - same_label) * F.relu(margin - distances)**2
    return loss.mean()

def prototypical_loss(support_embs, support_labels, query_embs, query_labels, n_way, k_shot):
    support_embs_flat = support_embs.view(-1, support_embs.size(-1))
    support_labels_flat = support_labels.view(-1)
    query_embs_flat = query_embs.view(-1, query_embs.size(-1))
    query_labels_flat = query_labels.view(-1)
    
    prototypes = compute_prototypes(support_embs_flat, support_labels_flat, n_way, k_shot)
    
    query_embs_flat = query_embs_flat.unsqueeze(1)
    prototypes = prototypes.unsqueeze(0)
    distances = -F.cosine_similarity(query_embs_flat, prototypes, dim=2)
    logits = -distances
    
    log_probs = F.log_softmax(logits, dim=1)
    loss = F.nll_loss(log_probs, query_labels_flat)
    preds = logits.argmax(dim=1)
    acc = (preds == query_labels_flat).float().mean()
    return loss, acc
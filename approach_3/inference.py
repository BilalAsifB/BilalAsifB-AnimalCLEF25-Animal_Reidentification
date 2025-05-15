import torch
import os
import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from .loss import compute_prototypes

def plot_tsne(support_embs, support_ids, query_embs, query_ids, species_map, species_list):
    all_embs = torch.cat([support_embs, query_embs], dim=0).cpu().numpy()
    all_ids = support_ids + query_ids
    all_species = [species_map.get(id_, 'Unknown') if id_ != 'query' else species_list[query_ids.index(id_) % len(species_list)] for id_ in all_ids]
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    embeddings_2d = tsne.fit_transform(all_embs)
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x=embeddings_2d[:, 0], y=embeddings_2d[:, 1], hue=all_species, style=['Support' if id_ in support_ids else 'Query' for id_ in all_ids], size=['Support' if id_ in support_ids else 'Query' for id_ in all_ids], palette='Set1')
    plt.title("t-SNE Visualization of Embeddings")
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.legend(title="Species / Type")
    plt.savefig(os.path.join('submissions', 'tsne_embeddings.png'))
    plt.close()

def inference(model, support_loader, query_loader, device, species_list, species_map, support_dataset, config):
    model.eval()
    support_embs = []
    support_ids = []
    with torch.no_grad():
        for support_imgs, support_labels, _, _ in support_loader:
            batch_size = support_imgs.size(0)
            max_len = support_imgs.size(1)
            imgs = support_imgs.view(batch_size * max_len, 3, config['image_size'][0], config['image_size'][1])
            imgs = imgs.to(device)
            embs = model(imgs, return_embeddings=True)
            embs = embs.view(batch_size, max_len, config['embedding_size']).mean(dim=1).cpu()
            support_embs.append(embs)
            for idx in range(batch_size):
                identity = support_dataset.identities[idx]
                support_ids.append(identity)
    support_embs = torch.cat(support_embs, dim=0)
    support_labels = torch.arange(len(support_ids))
    prototypes = compute_prototypes(support_embs.to(device), support_labels.to(device), len(support_ids), 1)

    final_predictions = {}
    all_query_ids = []
    max_probs_list = []
    with torch.no_grad():
        for imgs, img_ids, species in query_loader:
            imgs = imgs.to(device)
            query_embs = model(imgs, return_embeddings=True)
            if query_embs.dim() == 1:
                query_embs = query_embs.unsqueeze(0)
            distances = -torch.nn.functional.cosine_similarity(query_embs.unsqueeze(1), prototypes.unsqueeze(0), dim=2)
            distances = (distances - distances.min()) / (distances.max() - distances.min())
            logits = -distances
            probs = torch.nn.functional.softmax(logits, dim=1)
            max_probs, preds = probs.max(dim=1)
            max_probs_list.extend(max_probs.cpu().numpy())
            all_query_ids.extend(img_ids)
            for img_id, pred, max_prob in zip(img_ids, preds.cpu().numpy(), max_probs.cpu().numpy()):
                threshold = 0.7
                if max_prob < threshold:
                    final_predictions[img_id] = "new_individual"
                else:
                    final_predictions[img_id] = support_ids[pred]

    print(f"Prediction distribution: {Counter(final_predictions.values())}")

    n_samples = min(500, len(support_embs), len(all_query_ids))
    support_indices = random.sample(range(len(support_embs)), n_samples // 2)
    query_indices = random.sample(range(len(all_query_ids)), n_samples - (n_samples // 2))
    sampled_support_embs = support_embs[support_indices]
    sampled_support_ids = [support_ids[i] for i in support_indices]
    query_embs_list = []
    for imgs, _, _ in query_loader:
        imgs = imgs.to(device)
        embs = model(imgs, return_embeddings=True)
        query_embs_list.append(embs.cpu())
    query_embs = torch.cat(query_embs_list, dim=0)
    sampled_query_embs = query_embs[query_indices[:len(query_indices)]] if len(query_indices) <= len(query_embs) else query_embs[:len(query_indices)]
    sampled_query_ids = [f"query_{i}" for i in range(len(sampled_query_embs))]

    plot_tsne(sampled_support_embs, sampled_support_ids, sampled_query_embs, sampled_query_ids, species_map, species_list)
    return final_predictions, all_query_ids

def create_submission(query_dataset, predictions, submission_path):
    import pandas as pd
    sample_submission = pd.read_csv(os.path.join(query_dataset.metadata_dir, 'sample_submission.csv'))
    submission = sample_submission.copy()
    submission["Id"] = submission["image_id"].map(lambda x: predictions.get(str(x), "new_individual"))
    missing_ids = [x for x in submission["image_id"].astype(str) if x not in predictions]
    if missing_ids:
        print(f"Warning: {len(missing_ids)} image IDs missing from predictions")
    submission.to_csv(submission_path, index=False)
    print(f"✅ Submission file saved as {submission_path}.")
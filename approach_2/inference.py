import torch
import numpy as np
from .faiss_utils import build_faiss_index, search_faiss_index
from scripts.evaluate import create_submission
from evaluation.metrics import compute_baks, compute_baus
from sklearn.cluster import DBSCAN

def extract_embeddings_db(model, loader, device):
    model.eval()
    embeddings, labels, species = [], [], []
    with torch.no_grad():
        for imgs, labels_batch, species_labels in loader:
            imgs = imgs.to(device)
            feats = model.backbone(imgs)
            feats = model.pooling(feats).flatten(1)
            feats = model.embedding(feats)
            embeddings.append(feats.cpu().numpy())
            labels.extend(labels_batch.cpu().numpy())
            species.extend(species_labels)
    return np.vstack(embeddings), labels, species

def extract_embeddings_query(model, loader, device):
    model.eval()
    embeddings, image_ids, species = [], [], []
    with torch.no_grad():
        for imgs, img_ids, species_labels, _ in loader:
            imgs = imgs.to(device)
            feats = model.backbone(imgs)
            feats = model.pooling(feats).flatten(1)
            feats = model.embedding(feats)
            embeddings.append(feats.cpu().numpy())
            image_ids.extend(img_ids)
            species.extend(species_labels)
    return np.vstack(embeddings), image_ids, species

def one_shot_evaluation(model, database_loader, query_loaders, query_dataset, config):
    """Evaluate ArcFace model using FAISS and DBSCAN (CPU-only)."""
    device = torch.device('cpu')
    # Extract database embeddings
    db_embs, db_ids, db_species = extract_embeddings_db(model, database_loader, device)
    np.save(os.path.join(config['output_dir'], 'db_embeddings.npy'), db_embs)
    np.save(os.path.join(config['output_dir'], 'db_ids.npy'), np.array(db_ids))

    # Build FAISS index
    index = build_faiss_index(db_embs, config['embedding_size'])

    # Process queries
    full_query_embs = []
    full_query_ids = []
    full_query_species = []
    for loader in query_loaders:
        query_embs, query_ids, query_species = extract_embeddings_query(model, loader, device)
        full_query_embs.append(query_embs)
        full_query_ids.extend(query_ids)
        full_query_species.extend(query_species)

    full_query_embs = np.vstack(full_query_embs)

    # FAISS search
    D, I = search_faiss_index(index, full_query_embs, config['k_neighbors'])

    # Assign predictions
    final_predictions = {}
    new_inds_embs = []
    new_inds_ids = []
    for idx, (distances, neighbors) in enumerate(zip(D, I)):
        if distances[0] < config['new_threshold']:
            final_predictions[full_query_ids[idx]] = db_ids[neighbors[0]]
        else:
            new_inds_embs.append(full_query_embs[idx])
            new_inds_ids.append(full_query_ids[idx])

    # Cluster new individuals
    if new_inds_embs:
        new_inds_embs = np.vstack(new_inds_embs)
        clustering = DBSCAN(eps=0.6, min_samples=2, metric='euclidean').fit(new_inds_embs)
        cluster_labels = clustering.labels_
        cluster_to_newid = {}
        for img_id, cluster_id in zip(new_inds_ids, cluster_labels):
            if cluster_id == -1:
                final_predictions[img_id] = "new_individual"
            else:
                if cluster_id not in cluster_to_newid:
                    cluster_to_newid[cluster_id] = f"new_individual_{cluster_id}"
                final_predictions[img_id] = cluster_to_newid[cluster_id]

    # Compute metrics
    D, I = search_faiss_index(index, full_query_embs, 1)
    predicted_indices = I.flatten()
    query_labels = [config['label_map'][sp] for sp in full_query_species]

    baks_score = compute_baks(query_labels, full_query_species, db_ids, db_species, predicted_indices)
    baus_score = compute_baus(query_labels, full_query_species, db_ids, db_species, predicted_indices, D, threshold=config['new_threshold'])

    print(f"BAKS Score: {baks_score:.4f}, BAUS Score: {baus_score:.4f}")

    # Create submission
    sample_submission = pd.read_csv(os.path.join(config['data_root'], 'sample_submission.csv'))
    sample_submission["Id"] = sample_submission["image_id"].map(final_predictions).fillna("new_individual")
    sample_submission.to_csv(config['submission_path'], index=False)
    print(f"✅ Submission file saved as {config['submission_path']}.")
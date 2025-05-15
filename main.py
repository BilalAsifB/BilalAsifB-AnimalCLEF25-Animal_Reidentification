import argparse
import torch
from torchvision import transforms
from torch.utils.data import DataLoader, Subset
from scripts.utils import load_config
from scripts.preprocess import split_database
from approach1.dataset import AnimalCLEF2025, SiamesePairDataset
from approach1.model import SiameseNetwork
from approach1.train import train
from approach1.inference import one_shot_evaluation as siamese_evaluation
from approach2.dataset import AdvancedAnimalDataset, AdvancedTestDataset
from approach2.model import AnimalModel
from approach2.train import train_one_epoch
from approach2.inference import one_shot_evaluation as arcface_evaluation
from approach2.faiss_utils import build_faiss_index, search_faiss_index
from approach3.dataset import PretrainDataset, EpisodeDataset, QueryDataset, custom_collate
from approach3.model import ProtoNet
from approach3.train import pretrain_one_epoch, train_one_episode
from approach3.inference import inference as proto_inference, create_submission as proto_create_submission
import optuna
from torch.cuda.amp import GradScaler
import torch.optim as optim
import os
import numpy as np
from evaluation.compute_baks import compute_baks
from evaluation.compute_baus import compute_baus
from sklearn.cluster import DBSCAN

def extract_embeddings_db(model, loader, device, config):
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

def main(config_path, tune=False):
    config = load_config(config_path)
    device = torch.device('cpu')
    config['approach'] = config_path.split('/')[-1].split('_')[0]  # Extract approach from config file name

    if config['approach'] == 'siamese':
        transform = transforms.Compose([
            transforms.Resize(config['image_size']),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        dataset = AnimalCLEF2025(config['data_root'], transform=transform, load_label=True)
        train_idx = dataset.metadata[dataset.metadata['split'] == 'database'].index
        val_idx = dataset.metadata[dataset.metadata['split'] == 'validation'].index if 'validation' in dataset.metadata['split'].values else []
        train_dataset = dataset.get_subset(train_idx)
        val_dataset = dataset.get_subset(val_idx) if val_idx.size > 0 else []
        train_pair_dataset = SiamesePairDataset(train_dataset)
        val_pair_dataset = SiamesePairDataset(val_dataset) if val_dataset else []
        train_loader = DataLoader(train_pair_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=config['num_workers'])
        val_loader = DataLoader(val_pair_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=config['num_workers']) if val_pair_dataset else []
        database_dataset = dataset.get_subset(dataset.metadata['split'] == 'database')
        query_dataset = dataset.get_subset(dataset.metadata['split'] == 'query')
        database_loader = DataLoader(database_dataset, batch_size=1, shuffle=False, num_workers=config['num_workers'])
        query_loader = DataLoader(query_dataset, batch_size=1, shuffle=False, num_workers=config['num_workers'])

        model = SiameseNetwork(embedding_dim=config['embedding_dim']).to(device)
        train(model, train_loader, val_loader, config)
        model.load_state_dict(torch.load(config['model_path']))
        final_score = siamese_evaluation(model, database_loader, query_loader, query_dataset, config)
        print(f"Siamese Final Score: {final_score}")

    elif config['approach'] == 'arcface':
        train_transform = transforms.Compose([
            transforms.Resize(config['image_size']),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ])
        test_transform = transforms.Compose([
            transforms.Resize(config['image_size']),
            transforms.ToTensor()
        ])

        train_dirs = {
            "SeaTurtle": os.path.join(config['data_root'], 'images/SeaTurtleID2022/database'),
            "Salamander": os.path.join(config['data_root'], 'images/SalamanderID2025/database/images'),
            "Lynx": os.path.join(config['data_root'], 'images/LynxID2025/database')
        }
        train_datasets = [AdvancedAnimalDataset(train_dirs[sp], sp, transform=train_transform) for sp in config['label_map']]
        train_dataset = torch.utils.data.ConcatDataset(train_datasets)
        train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=config['num_workers'])

        if tune:
            def objective(trial):
                lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
                embedding_size = trial.suggest_categorical("embedding_size", [256, 512, 768])
                batch_size = trial.suggest_categorical("batch_size", [8, 16, 32])
                s = trial.suggest_float("arc_s", 15.0, 45.0)
                m = trial.suggest_float("arc_m", 0.3, 0.6)
                img_size = trial.suggest_categorical("img_size", [128, 150, 224])
                new_threshold = trial.suggest_float("new_threshold", 0.2, 0.7)
                dbscan_eps = trial.suggest_float("dbscan_eps", 0.4, 1.0)

                config['lr'] = lr
                config['embedding_size'] = embedding_size
                config['batch_size'] = batch_size
                config['image_size'] = [img_size, img_size]
                config['new_threshold'] = new_threshold
                config['arc_s'] = s
                config['arc_m'] = m

                train_transform = transforms.Compose([
                    transforms.Resize(config['image_size']),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor()
                ])
                test_transform = transforms.Compose([
                    transforms.Resize(config['image_size']),
                    transforms.ToTensor()
                ])

                train_datasets = [AdvancedAnimalDataset(train_dirs[sp], sp, transform=train_transform) for sp in config['label_map']]
                train_dataset = torch.utils.data.ConcatDataset(train_datasets)
                train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=config['num_workers'])
                val_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=config['num_workers'])

                model = AnimalModel(embedding_size=config['embedding_size'], num_classes=len(config['label_map']), arc_s=config['arc_s'], arc_m=config['arc_m']).to(device)
                optimizer = optim.Adam(model.parameters(), lr=config['lr'])
                scaler = GradScaler()

                for epoch in range(3):
                    train_one_epoch(model, train_loader, optimizer, device, scaler, config)

                db_embs, db_ids, db_species = extract_embeddings_db(model, train_loader, device)
                index = build_faiss_index(db_embs, config['embedding_size'])
                val_embs, val_ids, val_species = extract_embeddings_db(model, val_loader, device)
                D, I = search_faiss_index(index, val_embs, config['k_neighbors'])

                final_predictions = {}
                new_inds_embs, new_inds_ids = [], []
                for idx, (distances, neighbors) in enumerate(zip(D, I)):
                    if distances[0] < config['new_threshold']:
                        final_predictions[val_ids[idx]] = db_ids[neighbors[0]]
                    else:
                        new_inds_embs.append(val_embs[idx])
                        new_inds_ids.append(val_ids[idx])

                if new_inds_embs:
                    new_inds_embs = np.vstack(new_inds_embs)
                    clustering = DBSCAN(eps=dbscan_eps, min_samples=2, metric='euclidean').fit(new_inds_embs)
                    cluster_labels = clustering.labels_
                    cluster_to_newid = {}
                    for img_id, cluster_id in zip(new_inds_ids, cluster_labels):
                        if cluster_id == -1:
                            final_predictions[img_id] = "new_individual"
                        else:
                            if cluster_id not in cluster_to_newid:
                                cluster_to_newid[cluster_id] = f"new_individual_{cluster_id}"
                            final_predictions[img_id] = cluster_to_newid[cluster_id]

                D, I = search_faiss_index(index, val_embs, 1)
                predicted_indices = I.flatten()
                query_labels = [config['label_map'][sp] for sp in val_species]
                baus_score = compute_baus(query_labels, val_species, db_ids, db_species, predicted_indices, D, threshold=config['new_threshold'])
                return baus_score

            study = optuna.create_study(direction="maximize")
            study.optimize(objective, n_trials=20)
            best_params = study.best_trial.params
            config['lr'] = best_params['lr']
            config['embedding_size'] = best_params['embedding_size']
            config['batch_size'] = best_params['batch_size']
            config['image_size'] = [best_params['img_size'], best_params['img_size']]
            config['new_threshold'] = best_params['new_threshold']
            config['arc_s'] = best_params['arc_s']
            config['arc_m'] = best_params['arc_m']

        model = AnimalModel(embedding_size=config['embedding_size'], num_classes=len(config['label_map']), arc_s=config['arc_s'], arc_m=config['arc_m']).to(device)
        optimizer = optim.Adam(model.parameters(), lr=config['lr'])
        scaler = GradScaler()

        for epoch in range(config['num_epochs']):
            loss = train_one_epoch(model, train_loader, optimizer, device, scaler, config)
            print(f"ArcFace Epoch {epoch+1}/{config['num_epochs']}, Loss: {loss:.4f}")

        db_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=config['num_workers'])
        query_dirs = {
            "SeaTurtle": os.path.join(config['data_root'], 'images/SeaTurtleID2022/query'),
            "Salamander": os.path.join(config['data_root'], 'images/SalamanderID2025/query/images'),
            "Lynx": os.path.join(config['data_root'], 'images/LynxID2025/query')
        }
        query_loaders = []
        for species, qdir in query_dirs.items():
            if os.path.exists(qdir):
                dataset = AdvancedTestDataset(qdir, species, transform=test_transform)
                loader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=False, num_workers=config['num_workers'])
                query_loaders.append(loader)

        arcface_evaluation(model, db_loader, query_loaders, None, config)

    elif config['approach'] == 'proto':
        train_transform = transforms.Compose([
            transforms.Resize((config['image_size'][0] + 20, config['image_size'][1] + 20)),
            transforms.RandomCrop(config['image_size']),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x + torch.randn_like(x) * 0.02),
        ])
        test_transform = transforms.Compose([
            transforms.Resize(config['image_size']),
            transforms.ToTensor(),
        ])

        dataset = AnimalCLEF2025(config['data_root'], transform=None, load_label=True)
        database_dataset = dataset.get_subset(dataset.metadata['split'] == 'database')
        query_dataset = dataset.get_subset(dataset.metadata['split'] == 'query')

        pretrain_dataset = PretrainDataset(database_dataset, transform=train_transform)
        pretrain_loader = DataLoader(pretrain_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=config['num_workers'])

        model = ProtoNet(embedding_size=config['embedding_size'], num_classes=len(config['label_map'])).to(device)
        optimizer = optim.Adam(model.parameters(), lr=config['lr'])
        scaler = GradScaler()

        for epoch in range(config['pretrain_epochs']):
            loss, acc = pretrain_one_epoch(model, pretrain_loader, optimizer, scaler, device)
            print(f"Proto Pretrain Epoch {epoch+1}/{config['pretrain_epochs']}, Loss: {loss:.4f}, Accuracy: {acc:.4f}")

        train_dataset = EpisodeDataset(database_dataset, transform=train_transform, mode='train')
        train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=config['num_workers'], collate_fn=custom_collate)

        for epoch in range(config['epochs']):
            total_loss = 0
            total_acc = 0
            for support_imgs, support_labels, query_imgs, query_labels in train_loader:
                support_imgs = support_imgs.to(device)
                support_labels = support_labels.to(device)
                query_imgs = query_imgs.to(device)
                query_labels = query_labels.to(device)
                loss, acc = train_one_episode(model, support_imgs, support_labels, query_imgs, query_labels, optimizer, scaler, device, config)
                total_loss += loss
                total_acc += acc.item()
            avg_loss = total_loss / len(train_loader)
            avg_acc = total_acc / len(train_loader)
            print(f"Proto Epoch {epoch+1}/{config['epochs']}, Loss: {avg_loss:.4f}, Accuracy: {avg_acc:.4f}")

        support_dataset = EpisodeDataset(database_dataset, transform=test_transform, mode='test')
        support_loader = DataLoader(support_dataset, batch_size=1, shuffle=False, num_workers=config['num_workers'], collate_fn=custom_collate)
        query_loader = DataLoader(QueryDataset(query_dataset, transform=test_transform), batch_size=config['batch_size'], shuffle=False, num_workers=config['num_workers'])
        species_list = ["SeaTurtle", "Salamander", "Lynx"]
        final_predictions, all_query_ids = proto_inference(model, support_loader, query_loader, device, species_list, train_dataset.species_map, support_dataset, config)
        proto_create_submission(query_dataset, final_predictions, config['submission_path'])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run AnimalCLEF2025 approaches")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--tune", action="store_true", help="Run hyperparameter tuning (for ArcFace only)")
    args = parser.parse_args()
    main(args.config, args.tune)
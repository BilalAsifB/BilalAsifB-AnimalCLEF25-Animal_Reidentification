import os
import random
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

class PretrainDataset(Dataset):
    def __init__(self, dataset_database, transform=None):
        self.dataset_database = dataset_database
        self.transform = transform
        self.species_list = ["SeaTurtle", "Salamander", "Lynx"]
        self.label_map = CFG.label_map

    def __len__(self):
        return len(self.dataset_database)

    def __getitem__(self, idx):
        image, _ = self.dataset_database[idx]
        identity = self.dataset_database.metadata.iloc[idx]['identity']
        species = self.get_species(identity)
        label = self.label_map[species]
        if self.transform:
            image = self.transform(image)
        return image, label

    def get_species(self, identity):
        if 'LynxID2025' in identity:
            return 'Lynx'
        elif 'SalamanderID2025' in identity:
            return 'Salamander'
        elif 'SeaTurtleID2022' in identity:
            return 'SeaTurtle'
        return 'Unknown'

class EpisodeDataset(Dataset):
    def __init__(self, dataset_database, transform=None, mode='train'):
        self.dataset_database = dataset_database
        self.transform = transform
        self.mode = mode
        self.identity_to_indices = {}
        self.species_map = {}
        for idx, row in dataset_database.metadata.iterrows():
            identity = row['identity']
            species = self.get_species(identity)
            if identity not in self.identity_to_indices:
                self.identity_to_indices[identity] = []
            self.identity_to_indices[identity].append(idx)
            self.species_map[identity] = species
        self.identities = list(self.identity_to_indices.keys())
        self.weights = [1.0 / (max(1, len(indices)) ** 2) for indices in self.identity_to_indices.values()]

    def __len__(self):
        return 1000 if self.mode == 'train' else len(self.identities)

    def __getitem__(self, idx):
        selected_identities = random.choices(self.identities, weights=self.weights, k=CFG.n_way_train)
        support_images, query_images = [], []
        support_labels, query_labels = [], []

        for label, identity in enumerate(selected_identities):
            indices = self.identity_to_indices[identity]
            indices = random.sample(indices, min(len(indices), 5))
            if len(indices) == 1:
                idx = indices[0]
                img, _ = self.dataset_database[idx]
                if self.transform:
                    img = self.transform(img)
                support_images.append(img)
                support_labels.append(label)
                query_images.append(img)
                query_labels.append(label)
            else:
                sampled = random.sample(indices, min(len(indices), CFG.k_shot + CFG.q_query))
                support_idx = sampled[:CFG.k_shot]
                query_idx = sampled[CFG.k_shot:CFG.k_shot + CFG.q_query]
                for s_idx in support_idx:
                    img, _ = self.dataset_database[s_idx]
                    if self.transform:
                        img = self.transform(img)
                    support_images.append(img)
                    support_labels.append(label)
                for q_idx in query_idx:
                    img, _ = self.dataset_database[q_idx]
                    if self.transform:
                        img = self.transform(img)
                    query_images.append(img)
                    query_labels.append(label)

        return (torch.stack(support_images), torch.tensor(support_labels),
                torch.stack(query_images), torch.tensor(query_labels))

    def get_species(self, identity):
        if 'LynxID2025' in identity:
            return 'Lynx'
        elif 'SalamanderID2025' in identity:
            return 'Salamander'
        elif 'SeaTurtleID2022' in identity:
            return 'SeaTurtle'
        return 'Unknown'

class QueryDataset(Dataset):
    def __init__(self, dataset_query, transform=None):
        self.dataset_query = dataset_query
        self.transform = transform

    def __len__(self):
        return len(self.dataset_query)

    def __getitem__(self, idx):
        image, _ = self.dataset_query[idx]
        image_id = self.dataset_query.metadata.iloc[idx]['image_id']
        identity = str(self.dataset_query.metadata.iloc[idx]['identity']) if pd.notna(self.dataset_query.metadata.iloc[idx]['identity']) else 'Unknown'
        species = self.get_species(identity)
        if self.transform:
            image = self.transform(image)
        return image, str(image_id), species

    def get_species(self, identity):
        if 'LynxID2025' in identity:
            return 'Lynx'
        elif 'SalamanderID2025' in identity:
            return 'Salamander'
        elif 'SeaTurtleID2022' in identity:
            return 'SeaTurtle'
        return 'Unknown'

# Global CFG class (defined here for scope)
class CFG:
    seed = 42
    label_map = {"SeaTurtle": 0, "Salamander": 1, "Lynx": 2}
    img_size = 150
    n_way_train = 20
    k_shot = 1
    q_query = 5
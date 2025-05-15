import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class AnimalCLEF2025(Dataset):
    def __init__(self, root_dir, transform=None, load_label=True):
        self.root_dir = root_dir
        self.transform = transform
        self.load_label = load_label
        self.metadata = pd.read_csv(os.path.join(root_dir, 'metadata.csv'))
        self.metadata['path'] = self.metadata['path'].apply(lambda x: os.path.join(root_dir, x))

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        img_path = self.metadata.iloc[idx]['path']
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        label = self.metadata.iloc[idx]['identity'] if self.load_label else -1
        species = self.metadata.iloc[idx]['species']
        image_id = self.metadata.iloc[idx]['image_id']
        return image, label, species, image_id

    def get_subset(self, indices):
        subset = AnimalCLEF2025(self.root_dir, self.transform, self.load_label)
        subset.metadata = self.metadata.iloc[indices].reset_index(drop=True)
        return subset

class SiamesePairDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.metadata = dataset.metadata
        self.identities = self.metadata['identity'].unique()
        self.identity_to_indices = {identity: self.metadata[self.metadata['identity'] == identity].index.tolist() for identity in self.identities}

    def __len__(self):
        return len(self.dataset) * 2  # Generate pairs

    def __getitem__(self, idx):
        if idx % 2 == 0:  # Positive pair
            identity = random.choice(self.identities)
            idx1, idx2 = random.sample(self.identity_to_indices[identity], 2) if len(self.identity_to_indices[identity]) >= 2 else (self.identity_to_indices[identity][0], self.identity_to_indices[identity][0])
            label = 1
        else:  # Negative pair
            identity1, identity2 = random.sample(self.identities, 2)
            idx1 = random.choice(self.identity_to_indices[identity1])
            idx2 = random.choice(self.identity_to_indices[identity2])
            label = 0
        img1, _, _, _ = self.dataset[idx1]
        img2, _, _, _ = self.dataset[idx2]
        return img1, img2, torch.tensor(label, dtype=torch.float32)
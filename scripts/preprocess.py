import pandas as pd
from torch.utils.data import DataLoader
from torchvision import transforms
from approach1.dataset import AnimalCLEF2025

def split_database(data_root, transform, random_state=42):
    dataset = AnimalCLEF2025(data_root, transform=transform, load_label=True)
    metadata = dataset.metadata
    train_idx = metadata[metadata['split'] == 'database'].index
    val_idx = metadata[metadata['split'] == 'validation'].index
    train_dataset = dataset.get_subset(train_idx)
    val_dataset = dataset.get_subset(val_idx)
    return train_dataset, val_dataset
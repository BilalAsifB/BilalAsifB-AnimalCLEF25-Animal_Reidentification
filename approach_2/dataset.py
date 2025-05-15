import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class AdvancedAnimalDataset(Dataset):
    def __init__(self, root_dir, species_label, transform=None):
        self.root_dir = root_dir
        self.species_label = species_label
        self.transform = transform
        self.image_paths = []
        self.labels = []

        self.label = CFG.label_map[species_label]

        valid_exts = (".jpg", ".jpeg", ".png", ".bmp", ".tiff")
        for img_file in os.listdir(self.root_dir):
            if img_file.lower().endswith(valid_exts):
                img_path = os.path.join(self.root_dir, img_file)
                self.image_paths.append(img_path)
                self.labels.append(self.label)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label, self.species_label

class AdvancedTestDataset(Dataset):
    def __init__(self, root_dir, species, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.image_ids = []
        self.species = species
        self._load_data()

    def _load_data(self):
        for filename in os.listdir(self.root_dir):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                self.image_paths.append(os.path.join(self.root_dir, filename))
                self.image_ids.append(filename.split('.')[0])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        image_id = self.image_ids[idx]
        class_label = CFG.label_map[self.species]
        if self.transform:
            image = self.transform(image)
        return image, image_id, self.species, class_label

# Global CFG class (defined here for scope)
class CFG:
    seed = 42
    label_map = {"SeaTurtle": 0, "Salamander": 1, "Lynx": 2}
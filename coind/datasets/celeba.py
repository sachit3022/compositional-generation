from typing import Optional
import torch
from torch.utils.data import Dataset
from PIL import Image
import os
import csv
import numpy as np
from torchvision import transforms

class CSV:
    def __init__(self, headers, indices, data):
        self.header = headers
        self.index= indices
        self.data = data

def default_celeba_transform(split):
    if split == 'train':
        return transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

def blond_male_transform():
    return lambda x: (x[[20,9]]+1)//2
        
class CelebADataset(Dataset):
    def __init__(self, root, split='train',transforms=None,target_transform=None):
        self.root = root
        self.base_folder = 'celeba'
        self.img_dir = 'img_align_celeba'
        attr_file = 'list_attr_celeba.txt'
        partition_file = 'list_eval_partition.txt'
        if transforms is None:
            self.transform = default_celeba_transform(split)
        else:
            self.transform = transforms
        self.target_transform = target_transform
        # Load attributes and partitions using the _load_csv method
        self.attributes_csv = self._load_csv(attr_file, header=1)
        self.splits_csv = self._load_csv(partition_file)
        # Filter images based on the split
        mask = self.filter_data(split)

        self.filename = [self.splits_csv.index[i] for i in torch.squeeze(torch.nonzero(mask))]
        self.attributes = self.attributes_csv.data[mask]
        self.attr_names = self.attributes_csv.header

    def filter_data(self,split):
        partition_map = {'train': 0, 'val': 1, 'test': 2}               
        split_ = partition_map[split]
        mask = slice(None) if split_ is None else (self.splits_csv.data == split_).squeeze()
        return mask

    def _load_csv(self, filename: str, header: Optional[int] = None) -> CSV:
        with open(os.path.join(self.root, self.base_folder, filename)) as csv_file:
            data = list(csv.reader(csv_file, delimiter=" ", skipinitialspace=True))

        if header is not None:
            headers = data[header]
            data = data[header + 1 :]
        else:
            headers = []

        indices = [row[0] for row in data]
        data = [row[1:] for row in data]
        data_int = [list(map(int, i)) for i in data]
        return CSV(headers, indices, torch.tensor(data_int))
        
    def __len__(self) -> int:
        return len(self.attributes)
    
    def __getitem__(self, idx):
        img_name = os.path.join(f"{self.root}/{self.base_folder}/{self.img_dir}", self.filename[idx])
        image = Image.open(img_name)
        
        if self.transform:
            image = self.transform(image)
        
        attrs = self.attributes[idx]
        if self.target_transform:
            attrs = self.target_transform(attrs)

        return {"X": image, "label": attrs, "idx": int(self.filename[idx].split(".")[0])}

class BlondFemaleDataset(CelebADataset):
    def filter_data(self,split):
        partition_map = {'train': 0, 'val': 1, 'test': 2}               
        split_ = partition_map[split]
        mask = slice(None) if split_ is None else (self.splits_csv.data == split_).squeeze()
        if split == 'train':
            mask = mask & torch.logical_not(torch.logical_and(self.attributes_csv.data[:,9] == 1, self.attributes_csv.data[:,20] == -1))
        elif split == 'val':
            mask = mask & torch.logical_and(self.attributes_csv.data[:,9] == 1, self.attributes_csv.data[:,20] == -1)
        return mask

class AttrCelebALatent(CelebADataset):
    def __init__(self, celeba_dir,latent_dir,split='train'):
        self.root = celeba_dir
        self.base_folder = 'celeba'
        self.img_dir = 'img_align_celeba'
        attr_file = 'list_attr_celeba.txt'
        partition_file = 'list_eval_partition.txt'
        # Load attributes and partitions using the _load_csv method
        self.attributes_csv = self._load_csv(attr_file, header=1)
        self.splits_csv = self._load_csv(partition_file)
        # Filter images based on the split
        mask = self.filter_data(split)
        self.filename = [self.splits_csv.index[i] for i in torch.squeeze(torch.nonzero(mask))]
        self.attributes = self.attributes_csv.data[mask]
        self.attr_names = self.attributes_csv.header
        self.latent_dir = latent_dir

    
    def filter_data(self,split):
        raise NotImplementedError
        
    def __getitem__(self, index):
        self.images = np.load(self.latent_dir+"/{:06d}.npy".format(int(self.filename[index].split(".")[0])))
        self.labels = self.attributes[index]
        return {"X":torch.tensor(self.images), "label": (self.labels[[20,9]]+1)//2, 'label_null': torch.ones_like(self.labels[[20,9]])*2}
    def __len__(self):
        return len(self.attributes)

class CompositionalBlondMale(AttrCelebALatent):
    """
    Results from the CRM paper https://arxiv.org/pdf/2410.06303
    Method      Average Acc    Balanced Acc    Worst Group Acc
    ERM         87.0 (0.0)     59.3 (0.3)      4.0 (0.0)
    G-DRO       91.7 (0.3)     86.3 (0.7)      71.7 (0.9)
    LC          88.3 (0.3)     70.7 (0.7)      21.0 (2.1)
    sLA         88.3 (0.3)     71.0 (0.6)      21.3 (1.9)
    CRM         93.0 (0.0)     85.7 (0.3)      73.3 (1.8)
    CoInD       97.93          97.18           93.95
    """
    def filter_data(self,split):
        partition_map = {'train': 0, 'val': 1, 'test': 2}  
        split_ = partition_map[split]
        mask = slice(None) if split_ is None else (self.splits_csv.data == split_).squeeze()
        if split == 'train':
           mask = mask & torch.logical_not(torch.logical_and(self.attributes_csv.data[:,9] == 1, self.attributes_csv.data[:,20] == -1))
        elif split == 'val':
            mask = mask & torch.logical_and(self.attributes_csv.data[:,9] == 1, self.attributes_csv.data[:,20] == -1)
        return mask 
    
class BlondMale(AttrCelebALatent):
    """
    Results from the CRM paper https://arxiv.org/pdf/2410.06303
    Method      Average Acc    Balanced Acc    Worst Group Acc
    ERM         87.0 (0.0)     59.3 (0.3)      4.0 (0.0)
    G-DRO       91.7 (0.3)     86.3 (0.7)      71.7 (0.9)
    LC          88.3 (0.3)     70.7 (0.7)      21.0 (2.1)
    sLA         88.3 (0.3)     71.0 (0.6)      21.3 (1.9)
    CRM         93.0 (0.0)     85.7 (0.3)      73.3 (1.8)
    """
    def filter_data(self,split):
        partition_map = {'train': 0, 'val': 1, 'test': 2}               
        split_ = partition_map[split]
        mask = slice(None) if split_ is None else (self.splits_csv.data == split_).squeeze()
        return mask 
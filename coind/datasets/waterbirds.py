import os
import torch
import pandas as pd
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import Dataset, Subset

model_attributes = {
    'bert': {
        'feature_type': 'text'
    },
    'inception_v3': {
        'feature_type': 'image',
        'target_resolution': (299, 299),
        'flatten': False
    },
    'wideresnet50': {
        'feature_type': 'image',
        'target_resolution': (224, 224),
        'flatten': False
    },
    'resnet50': {
        'feature_type': 'image',
        'target_resolution': (224, 224),
        'flatten': False
    },
    'resnet34': {
        'feature_type': 'image',
        'target_resolution': None,
        'flatten': False
    },
    'raw_logistic_regression': {
        'feature_type': 'image',
        'target_resolution': None,
        'flatten': True,
    }
}

class ConfounderDataset(Dataset):
    def __init__(self, root_dir,
                 target_name, confounder_names,
                 augment_data=False,
                 model_type=None,split='train',transforms=None):
        raise NotImplementedError

    def __len__(self):
        return len(self.filename_array)

    def __getitem__(self, idx):
        y = self.y_array[idx]
        g = self.group_array[idx]

        if model_attributes[self.model_type]['feature_type']=='precomputed':
            x = self.features_mat[idx, :]
        else:
            img_filename = os.path.join(
                self.data_dir,
                self.filename_array[idx])
            img = Image.open(img_filename)
            if self.transforms:
                img = self.transforms(img)
            # Flatten if needed
            if model_attributes[self.model_type]['flatten']:
                assert img.dim()==3
                img = img.view(-1)
            x = img

        return {"X": x, "y": torch.tensor((y,g)), "idx": self.metadata_idx[idx]}

    def group_str(self, group_idx):
        y = group_idx // (self.n_groups/self.n_classes)
        c = group_idx % (self.n_groups//self.n_classes)

        group_name = f'{self.target_name} = {int(y)}'
        bin_str = format(int(c), f'0{self.n_confounders}b')[::-1]
        for attr_idx, attr_name in enumerate(self.confounder_names):
            group_name += f', {attr_name} = {bin_str[attr_idx]}'
        return group_name


class CUBDataset(ConfounderDataset):
    """
    CUB dataset (already cropped and centered).
    Note: metadata_df is one-indexed.
    """

    def __init__(self, root_dir,
                 target_name, confounder_names,
                 augment_data=False,
                 model_type=None,split='train',transforms=None):
        self.root_dir = root_dir
        self.target_name = target_name
        self.confounder_names = confounder_names
        self.model_type = model_type
        self.augment_data = augment_data

        self.data_dir = os.path.join(
            self.root_dir,
            'data',
            '_'.join([self.target_name] + self.confounder_names))

        if not os.path.exists(self.data_dir):
            raise ValueError(
                f'{self.data_dir} does not exist yet. Please generate the dataset first.')

        # Read in metadata
        self.metadata_df = pd.read_csv(
            os.path.join(self.data_dir, 'metadata.csv'))

        

        # Get the y values
        self.y_array = self.metadata_df['y'].values
        self.n_classes = 2

        # We only support one confounder for CUB for now
        self.confounder_array = self.metadata_df['place'].values
        self.n_confounders = 1
        # Map to groups
        self.n_groups = pow(2, 2)
        self.group_array = (self.y_array*(self.n_groups/2) + self.confounder_array).astype('int') % (self.n_groups//self.n_classes)

        # Extract filenames and splits
        self.filename_array = self.metadata_df['img_filename'].values
        self.metadata_idx = self.metadata_df.index
        self.split_array = self.metadata_df['split'].values
        self.split_dict = {
            'train': 0,
            'val': 1,
            'test': 2
        
        }

        #filter data
        mask = self.filter_data(split)
        self.y_array = self.y_array[mask]
        self.group_array = self.group_array[mask]
        self.filename_array = self.filename_array[mask]
        self.split_array = self.split_array[mask]
        self.metadata_idx = self.metadata_idx[mask]


        # Set transform
        self.transforms = transforms
    
    def filter_data(self,split):
        mask = self.split_array == self.split_dict[split]
        return mask


class AttrWaterbirdsLatent(CUBDataset):
    def __init__(self,root_dir,
                 target_name, confounder_names,
                 augment_data=False,
                 model_type=None,split='train',latent_dir=None):
        self.root_dir = root_dir
        self.target_name = target_name
        self.confounder_names = confounder_names
        self.model_type = model_type
        self.augment_data = augment_data

        self.data_dir = os.path.join(
            self.root_dir,
            'data',
            '_'.join([self.target_name] + self.confounder_names))

        if not os.path.exists(self.data_dir):
            raise ValueError(
                f'{self.data_dir} does not exist yet. Please generate the dataset first.')

        # Read in metadata
        self.metadata_df = pd.read_csv(
            os.path.join(self.data_dir, 'metadata.csv'))

        # Get the y values
        self.y_array = self.metadata_df['y'].values
        self.n_classes = 2

        # We only support one confounder for CUB for now
        self.confounder_array = self.metadata_df['place'].values
        self.metadata_idx = self.metadata_df.index
        self.n_confounders = 1
        # Map to groups
        self.n_groups = pow(2, 2)
        self.group_array = (self.y_array*(self.n_groups/2) + self.confounder_array).astype('int')

        # Extract filenames and splits
        self.filename_array = self.metadata_df['img_filename'].values
        self.split_array = self.metadata_df['split'].values
        self.split_dict = {
            'train': 0,
            'val': 1,
            'test': 2
        
        }


        #filter data
        mask = self.filter_data(split)
        self.y_array = self.y_array[mask]
        self.group_array = self.group_array[mask]% (self.n_groups//self.n_classes)
        self.filename_array = self.filename_array[mask]
        self.split_array = self.split_array[mask]
        self.metadata_idx = self.metadata_idx[mask]
        self.latent_dir = latent_dir

    
    def filter_data(self,split):
        raise NotImplementedError

    def __getitem__(self, idx):
        y = self.y_array[idx]
        g = self.group_array[idx]
        
        x = np.load(self.latent_dir+"/{:06d}.npy".format(self.metadata_idx[idx]))
        label = torch.tensor((y,g))
        return {"X": x, "label": label, "label_null": torch.ones_like(label)*2}

    def __len__(self):
        return len(self.y_array)

class Compositional01(AttrWaterbirdsLatent):
    def filter_data(self,split):
        mask = self.split_array == self.split_dict[split]
        if split == 'train':
            mask = mask & np.logical_not(np.logical_and(self.y_array == 0, self.group_array == 1))
        elif split == 'val':
            mask = mask & np.logical_and(self.y_array == 0, self.group_array == 1)
        return mask

class Full(AttrWaterbirdsLatent):
    def filter_data(self,split):
        mask = self.split_array == self.split_dict[split]
        return mask

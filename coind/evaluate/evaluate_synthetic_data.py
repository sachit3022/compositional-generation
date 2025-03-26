import sys
import os
from pathlib import Path
sys.path.append(str(Path(__file__).absolute().parent.parent))

from torch.utils.data import Dataset
import json
import pytorch_lightning as pl
import torch
from torch import nn
from torch.nn import functional as F
from torchvision import models
from torchmetrics import Metric
import argparse
from datasets.celeba import *
from datasets.sythetic import SytheticData
from torch.utils.data import ConcatDataset,WeightedRandomSampler,RandomSampler
from torchvision.models.resnet import ResNet18_Weights,ResNet50_Weights
from torch.utils.data import DataLoader
from metrics import DROMetrics
from collections import defaultdict,Counter
import matplotlib.pyplot as plt
import torchvision
import numpy as np
from torchmetrics.image.fid import FrechetInceptionDistance
import bisect
from torch.utils.data import ConcatDataset

class ConcatDatasetWithIndices(ConcatDataset):
    def __getitem__(self, idx):
        # Handle negative indices
        if idx < 0:
            if -idx > len(self):
                raise ValueError("absolute value of index should not exceed dataset length")
            idx = len(self) + idx

        # Determine which dataset this index falls into using cumulative sizes
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        
        # Get the sample from the appropriate dataset
        sample = self.datasets[dataset_idx][sample_idx]
        
        # Option 1: If your sample is a dictionary, add a new key
        if isinstance(sample, dict):
            sample['dataset_idx'] = dataset_idx
            return sample
        # Option 2: If your sample is a tuple (e.g., (data, label)), append the index
        elif isinstance(sample, tuple):
            return sample + (dataset_idx,)
        # Option 3: Otherwise, return a tuple with the sample and the dataset index
        else:
            return_dict  = {}
            for key in sample:
                return_dict[key] = sample[key]
            return_dict['dataset_idx'] = dataset_idx
            return return_dict

def calculate_sample_weights(labels):
    # Convert labels to tuples for easy counting
    label_tuples = [tuple(label) for label in labels]
    
    # Count occurrences of each unique label combination
    group_counts = Counter(label_tuples)

    #unique code for balancing first balance all the groups (1,0,*)(0,1,*)(1,1,*)(0,0,*) should be equal and if (1,0,*) has (1,0,1) (1,0,0) then they should be balanced
    group2_dict = defaultdict(list)
    for idx, label in enumerate(labels):
        group_key = (label[0], label[1])
        group2_dict[group_key].append(idx)
    
    num_groups = len(group2_dict)
    balanced_group_weights = {}
    
    # For each group defined by the first two label values,
    # further split by the full label (which may differ in the 3rd element)
    for group_key, indices in group2_dict.items():
        # Count how many samples there are per full label within this group
        subgroup_counts = Counter(tuple(labels[i]) for i in indices)
        num_subgroups = len(subgroup_counts)
        for label_tuple, count in subgroup_counts.items():
            # Each group gets equal overall weight (1/num_groups)
            # and within the group each subgroup gets an equal share (1/num_subgroups).
            # Then each sample in a subgroup receives the share divided by its count.
            balanced_weight = (1.0 / num_groups) * (1.0 / num_subgroups) / count
            balanced_group_weights[label_tuple] = balanced_weight

    sample_weights = [balanced_group_weights[tuple(label)] for label in labels]

    
    return torch.tensor(sample_weights)


def unseen_from_synthetic_and_rest_all_from_real(labels):
    # Convert labels to tuples for easy counting
    label_tuples = [tuple(label) for label in labels]
    
    # Count occurrences of each unique label combination
    group_counts = Counter(label_tuples)

    #unique code for balancing first balance all the groups (1,0,*)(0,1,*)(1,1,*)(0,0,*) should be equal and if (1,0,*) has (1,0,1) (1,0,0) then they should be balanced
    #(0,1,0) _> 25% and (1,1,1) -> 25% and (1,0,1) -> 25% and (0,0,1) -> 25%
    counter_dict = defaultdict(int)
    for idx, label in enumerate(labels):
        group_key = (label[0], label[1],label[2])
        counter_dict[group_key] += 1
    
    sample_weights =[]
    for idx, label in enumerate(labels):
        group_key = (label[0], label[1],label[2])
        if group_key in [(0,1,0),(1,1,1),(1,0,1),(0,0,1)]:
            sample_weights.append(1.0/counter_dict[group_key])
        else:
            sample_weights.append(0)

    
    return torch.tensor(sample_weights)

if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    

    parser = argparse.ArgumentParser()
    parser.add_argument("-s","--sythetic_data_path", type=str,required=True)
    parser.add_argument("-og","--original_data_path",type=str,required=True)
    parser.add_argument("-on","--train_on",type=str,choices=["synthetic","original","both"],required=True)
    
    args = parser.parse_args()

    train_transform = default_celeba_transform('train')
    val_transform = default_celeba_transform('val')
    
    sythetic_data = SytheticData(args.sythetic_data_path,transform=train_transform)

    #devide sythetic data into train and val
    sythetic_data_train,sythetic_data_val = torch.utils.data.random_split(sythetic_data,[int(len(sythetic_data)*0.8),len(sythetic_data)-int(len(sythetic_data)*0.8)])
    sythetic_data_val.dataset.transform = val_transform
    original_data_train = BlondFemaleDataset(args.original_data_path, split='train',transforms=val_transform,target_transform=blond_male_transform())
    original_data_val = CelebADataset(args.original_data_path, split='val',transforms=val_transform,target_transform=blond_male_transform())

    sampler = None
    if args.train_on == "synthetic":
        train_data = sythetic_data_train
        sampler = RandomSampler(train_data)
    elif args.train_on == "original":
        train_data = original_data_train
        sampler = RandomSampler(train_data)
    elif args.train_on == "both":
        train_data = ConcatDatasetWithIndices([sythetic_data_train,original_data_train]) 
        sampler = WeightedRandomSampler(unseen_from_synthetic_and_rest_all_from_real( [sythetic_data_train.dataset.metadata[i]['query']+[0] for i in sythetic_data_train.indices ] + [x+[1] for x in ((train_data.datasets[1].attributes[:,[20,9]]+1)//2).tolist()]) ,len(train_data))
    else:
        raise ValueError(f"Unknown train_on: {args.train_on}")


    
    
    model = models.resnet50(weights=ResNet50_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, 2)
    
    
    
    model.to(device)
    batch_size = 1024

    train_dataloader = DataLoader(train_data,batch_size=batch_size,num_workers=4,persistent_workers=True, sampler=sampler)
    val_dataloader = DataLoader(original_data_val,batch_size=batch_size,shuffle=False,num_workers=4,persistent_workers=True)
    syn_val_dataloader = DataLoader(sythetic_data_val,batch_size=batch_size,shuffle=False,num_workers=4,persistent_workers=True)

    epoch = 15
    train_acc = DROMetrics()
    val_acc = DROMetrics()
    val_acc_syn = DROMetrics()

    optimizer = torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,epoch)

    #counter to count the number of times the model is trained on the synthetic data
    for i  in range(epoch):
        counter = Counter()
        model.train()
        for batch in train_dataloader:
            x,y = batch['X'].to(device),batch['label'].to(device)
            if 'dataset_idx' in batch:
                batch_dataset_idx = batch['dataset_idx']
                #make a dict of batch_dataset_idx,y[:,0],y[:,1] and increment the counter
                for idx,y_0,y_1 in zip(batch_dataset_idx,y[:,0],y[:,1]):
                    counter[(idx.item(),y_0.item(),y_1.item())] += 1
            logits = model(x)
            loss = F.cross_entropy(logits,y[:,1])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_acc.update(logits,y[:,1],y[:,0])
        scheduler.step()

            

        print(f"Epoch {i} Train Accuracy: {train_acc.compute()}")
        print(f"Epoch {i} Train Counter: {counter}")
        train_acc.reset()
        with torch.no_grad():
            model.eval()
            for batch_val in val_dataloader:
                x,y = batch_val['X'].to(device),batch_val['label'].to(device)
                logits = model(x)
                val_acc.update(logits,y[:,1],y[:,0])
        print(f"Epoch {i} Val Accuracy: {val_acc.compute()}")
        val_acc.reset()

        with torch.no_grad():
            model.eval()
            for batch_val in syn_val_dataloader:
                x,y = batch_val['X'].to(device),batch_val['label'].to(device)
                logits = model(x)
                val_acc_syn.update(logits,y[:,1],y[:,0])
        print(f"Epoch {i} Syn Val Accuracy: {val_acc_syn.compute()}")
        val_acc_syn.reset()



    ### FID Score
    fid_transform= transforms.Compose([
            transforms.Resize((299, 299)),
            transforms.PILToTensor()
            ])
    sythetic_data = SytheticData(args.sythetic_data_path,transform=fid_transform)
    original_data_val = CelebADataset(args.original_data_path, split='val',transforms=fid_transform,target_transform=blond_male_transform())

    train_loader = DataLoader(sythetic_data, batch_size=batch_size, shuffle=False, num_workers=4)
    val_loader = DataLoader(original_data_val, batch_size=batch_size, shuffle=False, num_workers=4)
    
    for y_0,y_1 in [(0,0),(0,1),(1,0),(1,1)]:
        fid = FrechetInceptionDistance(feature=768).to(device)
        for batch in train_loader:
            x = batch['X'].to(device)[torch.logical_and(batch['label'][:,1] == y_1,batch['label'][:,0] == y_0)]
            fid.update(x, real=False)

        for batch in val_loader:
            x = batch['X'].to(device)[torch.logical_and(batch['label'][:,1] == y_1,batch['label'][:,0] == y_0)]
            fid.update(x, real=True)
            
        print(f"{y_0},{y_1} FID: {fid.compute()}")



        


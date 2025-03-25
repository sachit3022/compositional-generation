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
from datasets.waterbirds import *
from datasets.sythetic import SytheticData
from torch.utils.data import ConcatDataset,WeightedRandomSampler,RandomSampler
from torch.utils.data import DataLoader
from metrics import DROMetrics
from collections import Counter


def calculate_sample_weights(labels):
    # Convert labels to tuples for easy counting
    label_tuples = [tuple(label) for label in labels]
    
    # Count occurrences of each unique label combination
    group_counts = Counter(label_tuples)
    
    # Calculate weights for each group
    total_samples = len(labels)
    group_weights = {group: total_samples / count for group, count in group_counts.items()}
    
    # Assign weights to each sample based on its group
    sample_weights = [group_weights[tuple(label)] for label in labels]
    
    return torch.tensor(sample_weights)

if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    

    parser = argparse.ArgumentParser()
    parser.add_argument("-s","--sythetic_data_path", type=str,required=True)
    parser.add_argument("-on","--train_on",type=str,choices=["synthetic","original","both"],default="synthetic")
    
    args = parser.parse_args()
    

    og_dataset_settings = {'root_dir':"/research/hal-gaudisac/Diffusion/controllable-generation/",
        'target_name': 'waterbird_complete95',
        'confounder_names': ['forest2water2'],
        "augment_data": None,
        'model_type': 'resnet50',
    }
    
    sythetic_data = SytheticData(args.sythetic_data_path,transform=None)
    original_data_train = Compositional01(latent_dir= "/research/hal-gaudisac/Diffusion/compositional-generation/data/waterbirds/clip_train_features",split= 'train',**og_dataset_settings)
    original_data_val = Full(latent_dir= "/research/hal-gaudisac/Diffusion/compositional-generation/data/waterbirds/clip_val_features",split= 'val',**og_dataset_settings)

    sampler = None
    if args.train_on == "synthetic":
        train_data = sythetic_data
        sampler = RandomSampler(train_data)
    elif args.train_on == "original":
        train_data = original_data_train
        sampler = RandomSampler(train_data)
    elif args.train_on == "both":
        train_data = ConcatDataset([sythetic_data,original_data_train]) 
        sampler = WeightedRandomSampler(calculate_sample_weights([x['query'] for x in train_data.datasets[0].metadata] + [[x,y] for x,y in zip(train_data.datasets[1].y_array,train_data.datasets[1].group_array)]),len(train_data))
    else:
        raise ValueError(f"Unknown train_on: {args.train_on}")


    
    ## MLP model   
    model = nn.Sequential(
        nn.Linear(768,512),
        nn.SiLU(),
        nn.Linear(512,2)
    )

    ########## Training the model ################ ( this should not change )
    
    
    
    model.to(device)
    batch_size = 64

    train_dataloader = DataLoader(train_data,batch_size=batch_size,num_workers=4,persistent_workers=True, sampler=sampler)
    val_dataloader = DataLoader(original_data_val,batch_size=batch_size,shuffle=False,num_workers=4,persistent_workers=True)


    epoch = 30
    train_acc = DROMetrics()
    val_acc = DROMetrics()
    optimizer = torch.optim.Adam(model.parameters(),lr=3e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,epoch)

    for i  in range(epoch):
        model.train()
        for batch in train_dataloader:
            x,y = batch['X'].to(device),batch['label'].to(device)
            logits = model(x)
            loss = F.cross_entropy(logits,y[:,0])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            train_acc.update(logits,y[:,0],y[:,1])

        print(f"Epoch {i} Train Accuracy: {train_acc.compute()}")
        train_acc.reset()
        with torch.no_grad():
            model.eval()
            for batch in val_dataloader:
                x,y = batch['X'].to(device),batch['label'].to(device)
                logits = model(x)
                val_acc.update(logits,y[:,0],y[:,1])
        print(f"Epoch {i} Val Accuracy: {val_acc.compute()}")
        val_acc.reset()


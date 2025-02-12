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
from torchvision.models.resnet import ResNet18_Weights
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
    parser.add_argument("-og","--original_data_path",type=str,required=True)
    parser.add_argument("-on","--train_on",type=str,choices=["synthetic","original","both"],required=True)
    
    args = parser.parse_args()

    train_transform = default_celeba_transform('train')
    val_transform = default_celeba_transform('val')
    sythetic_data = SytheticData(args.sythetic_data_path,transform=train_transform)
    # original_data_train = CompositionalBlondMale(args.original_data_path, latent_dir = '/research/hal-gaudisac/Diffusion/compositional-generation/data/celeba/vae_train_features', split='train')
    # original_data_val = BlondMale(args.original_data_path, latent_dir = '/research/hal-gaudisac/Diffusion/compositional-generation/data/celeba/vae_val_features', split='val')

    original_data_train = BlondFemaleDataset(args.original_data_path, split='train',transforms=train_transform,target_transform=blond_male_transform())
    original_data_val = BlondFemaleDataset(args.original_data_path, split='train',transforms=val_transform,target_transform=blond_male_transform())

    sampler = None
    if args.train_on == "synthetic":
        train_data = sythetic_data
        sampler = RandomSampler(train_data)
    elif args.train_on == "original":
        train_data = original_data_train
        sampler = RandomSampler(train_data)
    elif args.train_on == "both":
        train_data = ConcatDataset([sythetic_data,original_data_train]) 
        sampler = WeightedRandomSampler(calculate_sample_weights([x['query'] for x in train_data.datasets[0].metadata] + ((train_data.datasets[1].attributes[:,[20,9]]+1)//2).tolist()),len(train_data))
    else:
        raise ValueError(f"Unknown train_on: {args.train_on}")


    
    
    model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
    #model.conv1 = nn.Conv2d(16, 64, kernel_size=7, stride=2, padding=3,bias=False)
    model.fc = nn.Linear(model.fc.in_features, 2)
    
    
    
    model.to(device)
    batch_size = 1024

    train_dataloader = DataLoader(train_data,batch_size=batch_size,num_workers=4,persistent_workers=True, sampler=sampler)
    val_dataloader = DataLoader(original_data_val,batch_size=batch_size,shuffle=False,num_workers=4,persistent_workers=True)
    k=10020
    for batch in val_dataloader:
        x= batch['X']
        for j in range(len(x)):
            img = x[j].permute(1,2,0).cpu().numpy()
            img = (img - img.min())/(img.max()-img.min())
            Image.fromarray((img*255).astype(np.uint8)).save(f"samples/fid_og/sample_{k}.png")
            k+=1


    # epoch = 30
    # train_acc = DROMetrics()
    # val_acc = DROMetrics()
    # optimizer = torch.optim.Adam(model.parameters(),lr=3e-4)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,epoch)

    # for i  in range(epoch):
    #     model.train()
    #     for batch in train_dataloader:
    #         x,y = batch['X'].to(device),batch['label'].to(device)
    #         logits = model(x)
    #         loss = F.cross_entropy(logits,y[:,1])
    #         optimizer.zero_grad()
    #         loss.backward()
    #         optimizer.step()
    #         scheduler.step()
    #         train_acc.update(logits,y[:,1],y[:,0])

    #     print(f"Epoch {i} Train Accuracy: {train_acc.compute()}")
    #     train_acc.reset()
    #     with torch.no_grad():
    #         model.eval()
    #         for batch in val_dataloader:
    #             x,y = batch['X'].to(device),batch['label'].to(device)
    #             logits = model(x)
    #             val_acc.update(logits,y[:,1],y[:,0])
    #             if i == 0:
    #                 for j in range(10):
    #                     img = x[j].permute(1,2,0).cpu().numpy()
    #                     img = (img - img.min())/(img.max()-img.min())
    #                     Image.fromarray((img*255).astype(np.uint8)).save(f"samples/sample_{j}.png")


    #     print(f"Epoch {i} Val Accuracy: {val_acc.compute()}")
    #     val_acc.reset()


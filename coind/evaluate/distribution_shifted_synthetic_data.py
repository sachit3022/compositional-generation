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
from collections import Counter, defaultdict
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

# def calculate_sample_weights(labels):
#     # Convert labels to tuples for easy counting
#     label_tuples = [tuple(label) for label in labels]
    
#     # Count occurrences of each unique label combination
#     group_counts = Counter(label_tuples)
    
#     # Calculate weights for each group
#     total_samples = len(labels)
#     group_weights = {group: total_samples / count for group, count in group_counts.items()}
    
#     # Assign weights to each sample based on its group
#     sample_weights = [group_weights[tuple(label)] for label in labels]
    
#     return torch.tensor(sample_weights)

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


def gaussian_kernel(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    """
    Compute the multi-scale Gaussian kernel between source and target.
    """
    n_samples = int(source.size(0)) + int(target.size(0))
    total = torch.cat([source, target], dim=0)
    
    # Compute pairwise squared L2 distances
    total0 = total.unsqueeze(0).expand(total.size(0), total.size(0), total.size(1))
    total1 = total.unsqueeze(1).expand(total.size(0), total.size(0), total.size(1))
    L2_distance = ((total0 - total1) ** 2).sum(2)
    
    if fix_sigma:
        bandwidth = fix_sigma
    else:
        # Use the median heuristic
        bandwidth = torch.sum(L2_distance.data) / (n_samples**2 - n_samples)
    
    bandwidth_list = [bandwidth * (kernel_mul ** (i - kernel_num // 2)) for i in range(kernel_num)]
    kernel_vals = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
    
    return sum(kernel_vals)

def mmd_loss(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    """
    Compute the MMD loss between source and target features.
    """
    n_source = source.size(0)
    n_target = target.size(0)
    kernels = gaussian_kernel(source, target, kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)
    
    # Slicing the combined kernel matrix into four parts:
    XX = kernels[:n_source, :n_source]       # Source-Source
    YY = kernels[n_source:, n_source:]       # Target-Target
    XY = kernels[:n_source, n_source:]       # Source-Target
    YX = kernels[n_source:, :n_source]       # Target-Source
    
    loss = XX.mean() + YY.mean() - XY.mean() - YX.mean()
    return loss


def train_step(x,y,s, feature_extractor, classifier, optimizer, criterion, mmd_weight=1.0):
    """
    Performs one training step:
      - Computes the classification loss on batch['y'].
      - Computes the MMD loss to enforce invariance across dataset domains.
    
    Args:
        batch (dict): Dictionary with keys 'x', 'y', and 'dataset_idx'.
        feature_extractor (nn.Module): Network to extract features from input.
        classifier (nn.Module): Classifier that predicts classes based on features.
        optimizer (Optimizer): Optimizer for updating parameters.
        criterion (nn.Module): Classification loss (e.g., CrossEntropyLoss).
        mmd_weight (float): Weight to balance the MMD loss.
        
    Returns:
        tuple: (total loss, classification loss, MMD loss)
    """
    optimizer.zero_grad()

    # Forward pass through the feature extractor and classifier.
    features = feature_extractor(x)
    predictions = classifier(features)
    
    # Compute classification loss.
    cls_loss = criterion(predictions, y)
    
    # Separate features based on the dataset index.
    domain0_idx = (s == 0).nonzero(as_tuple=True)[0]
    domain1_idx = (s == 1).nonzero(as_tuple=True)[0]
    
    # Compute MMD loss only if both domains are present in the batch.
    if domain0_idx.numel() > 0 and domain1_idx.numel() > 0:
        features_domain0 = features[domain0_idx]
        features_domain1 = features[domain1_idx]
        domain_loss = mmd_loss(features_domain0, features_domain1)
    else:
        domain_loss = torch.tensor(0.0, device=x.device)
    
    total_loss = cls_loss + mmd_weight * domain_loss
    total_loss.backward()
    optimizer.step()
    
    return total_loss.item(), cls_loss.item(), domain_loss.item()

def compute_irm_penalty_split(logits, y):
    """
    Computes the IRM penalty using a split-batch technique for an unbiased estimate.
    
    If the batch is large enough (≥2 samples), the batch is split into two halves,
    and the penalty is defined as the dot product of the gradients computed on each half.
    
    Args:
        logits (Tensor): Logits for an environment of shape [n, num_classes].
        y (Tensor): True labels for the environment of shape [n].
    
    Returns:
        Tensor: A scalar IRM penalty.
    """
    n = logits.shape[0]
    if n < 2:
        # Fallback: if too few samples, compute penalty using the standard method.
        scale = torch.tensor(1.0, requires_grad=True, device=logits.device)
        loss = F.cross_entropy(logits * scale, y)
        grad = torch.autograd.grad(loss, [scale], create_graph=True)[0]
        return grad.pow(2).sum()
    else:
        # Split the batch into two halves
        half = n // 2
        scale = torch.tensor(1.0, requires_grad=True, device=logits.device)
        
        # Compute loss on first half
        loss1 = F.cross_entropy(logits[:half] * scale, y[:half])
        # Compute loss on second half
        loss2 = F.cross_entropy(logits[half:] * scale, y[half:])
        
        # Compute gradients w.r.t. the dummy scale parameter
        grad1 = torch.autograd.grad(loss1, [scale], create_graph=True)[0]
        grad2 = torch.autograd.grad(loss2, [scale], create_graph=True)[0]
        
        # Dot product of the gradients as the penalty (unbiased estimate of squared norm)
        penalty = (grad1 * grad2).sum()
        return penalty



def train_step_irm(x,y,s, feature_extractor, classifier, optimizer, global_step, penalty_anneal_iters, irm_weight):
    """
    Performs one training step using IRM with the following improvements:
      - Per-environment classification loss computation.
      - Unbiased IRM penalty computed via split-batch technique.
      - Penalty annealing: full IRM penalty weight is applied only after penalty_anneal_iters.
    
    Args:
        batch (dict): Contains 'x', 'y', and 'dataset_idx'.
        feature_extractor (nn.Module): Network extracting features from input.
        classifier (nn.Module): Classifier producing logits from features.
        optimizer (Optimizer): Optimizer for parameter updates.
        global_step (int): The current training iteration.
        penalty_anneal_iters (int): Number of iterations to wait before applying the full penalty.
        irm_weight (float): The full IRM penalty weight to use after annealing.
        
    Returns:
        tuple: (total_loss, mean_classification_loss, mean_irm_penalty)
    """
    optimizer.zero_grad()

    
    features = feature_extractor(x)
    env_losses = []
    env_penalties = []
    
    # Process each environment separately
    for env in torch.unique(s,dim=0):
        if len(env.shape) ==0:
            mask = (s == env)
        else:
            mask = (s == env).all(dim=1)
            
        if mask.sum() == 0:
            continue
        
        features_env = features[mask]
        y_env = y[mask]
        logits_env = classifier(features_env)
        
        # Compute classification loss for the environment
        loss_env = F.cross_entropy(logits_env, y_env)
        env_losses.append(loss_env)
        
        # Compute the IRM penalty using the split-batch approach
        penalty = compute_irm_penalty_split(logits_env, y_env)
        env_penalties.append(penalty)
    
    # Average over environments
    if env_losses:
        mean_loss = torch.stack(env_losses).mean()
    else:
        mean_loss = torch.tensor(0.0, device=x.device)
    
    if env_penalties:
        mean_penalty = torch.stack(env_penalties).mean()
    else:
        mean_penalty = torch.tensor(0.0, device=x.device)
    
    # Penalty annealing: before a certain number of iterations, use a lower penalty weight.
    if global_step < penalty_anneal_iters:
        penalty_weight = 1.0
    else:
        penalty_weight = irm_weight
    
    total_loss = mean_loss + penalty_weight * mean_penalty
    
    # Optionally: if using a very large penalty weight, you might rescale the loss
    # to mitigate sudden gradient spikes (or reset optimizer state).
    # For example:
    # if penalty_weight > 1.0:
    #     total_loss /= penalty_weight
    
    total_loss.backward()
    optimizer.step()
    
    return total_loss.item(), mean_loss.item(), mean_penalty.item()


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
        sampler = WeightedRandomSampler(calculate_sample_weights( [sythetic_data_train.dataset.metadata[i]['query']+[0] for i in sythetic_data_train.indices ] + [x+[1] for x in ((train_data.datasets[1].attributes[:,[20,9]]+1)//2).tolist()]) ,len(train_data))
    else:
        raise ValueError(f"Unknown train_on: {args.train_on}")


    
    
    feature_extractor = models.resnet50(weights=ResNet50_Weights.DEFAULT)
    classifier = nn.Linear(feature_extractor.fc.in_features,2)
    feature_extractor.fc = nn.Identity()
    
    
    
    feature_extractor.to(device)
    classifier.to(device)

    batch_size = 1024

    train_dataloader = DataLoader(train_data,batch_size=batch_size,num_workers=4,persistent_workers=True, sampler=sampler)
    val_dataloader = DataLoader(original_data_val,batch_size=batch_size,shuffle=False,num_workers=4,persistent_workers=True)
    syn_val_dataloader = DataLoader(sythetic_data_val,batch_size=batch_size,shuffle=False,num_workers=4,persistent_workers=True)

    epoch = 10
    train_acc = DROMetrics()
    val_acc = DROMetrics()
    val_acc_syn = DROMetrics()

    optimizer = torch.optim.AdamW(list(feature_extractor.parameters()) + list(classifier.parameters()),lr=3e-4,weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,epoch)
    criterion = nn.CrossEntropyLoss()

    global_step =0
    irm_weight = 1000.0      # Full penalty weight after annealing
    penalty_anneal_iters = 500  # Number of steps before full penalty is applied
    
    for i  in range(epoch):
        counter = Counter()
        feature_extractor.train()
        classifier.train()

        for batch in train_dataloader:
            x,y = batch['X'].to(device),batch['label'].to(device)
            batch_dataset_idx = batch['dataset_idx'].to(device)
            #make a dict of batch_dataset_idx,y[:,0],y[:,1] and increment the counter
            for idx,y_0,y_1 in zip(batch_dataset_idx,y[:,0],y[:,1]):
                counter[(idx.item(),y_0.item(),y_1.item())] += 1    
            #total_loss, cls_loss, mmd_loss_val = train_step(x,y[:,1],y[:,0], feature_extractor, classifier, optimizer, criterion)
            #torch.hstack([y[:,0].unsqueeze(dim=1),batch_dataset_idx.unsqueeze(dim=1)])
            total_loss, cls_loss, irm_penalty_val = train_step_irm(x,y[:,1],batch_dataset_idx, feature_extractor, classifier, optimizer,global_step, penalty_anneal_iters, irm_weight)
            global_step+=1
            with torch.no_grad():
                logits = classifier(feature_extractor(x))
                train_acc.update(logits,y[:,1],y[:,0])
        
        scheduler.step()
        print(f"Total Loss: {total_loss:.4f}, Classification Loss: {cls_loss:.4f}, MMD Loss: {irm_penalty_val:.4f}")
        print(f"Epoch {i} Train Accuracy: {train_acc.compute()}")
        print(f"Epoch {i} Train Counter: {counter}")
        
        train_acc.reset()
        with torch.no_grad():
            feature_extractor.eval()
            classifier.eval()

            for batch_val in val_dataloader:
                x,y = batch_val['X'].to(device),batch_val['label'].to(device)
                logits = classifier(feature_extractor(x))
                val_acc.update(logits,y[:,1],y[:,0])
        print(f"Epoch {i} Val Accuracy: {val_acc.compute()}")
        val_acc.reset()

        with torch.no_grad():
            feature_extractor.eval()
            classifier.eval()
            for batch_val in syn_val_dataloader:
                x,y = batch_val['X'].to(device),batch_val['label'].to(device)
                logits = classifier(feature_extractor(x))
                val_acc_syn.update(logits,y[:,1],y[:,0])
        print(f"Epoch {i} Syn Val Accuracy: {val_acc_syn.compute()}")
        val_acc_syn.reset()





        


import torch
from torch import nn
from torch.nn import functional as F
import math


class MultiLabelClassifier(nn.Module):
    def __init__(self,base_model:nn.Module,num_classes_per_label:list[int]):
        super().__init__()
        self.model = base_model
        dim_features = self.model.fc.in_features
        self.model.fc = nn.Identity()
        self.classifier_heads = nn.ModuleList([nn.Linear(dim_features,num_class) for num_class in num_classes_per_label])
        self.num_classes_per_label = num_classes_per_label
        
    def forward(self,x):
        x = self.model(x)
        return [head(x) for head in self.classifier_heads]

class VAEClassifer(nn.Module):
    def __init__(self,num_classes_per_label:list[int],input_dim:int=(16,8,8)):
        super().__init__()

        input_dim = math.prod(input_dim)
        hidden_dim = input_dim//2
        self.mlp = nn.Sequential(*[
            nn.Linear(input_dim,hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim,hidden_dim),        
            nn.GELU(),
        ])
        self.classifier_heads = nn.ModuleList([nn.Linear(hidden_dim,num_class) for num_class in num_classes_per_label])
        self.num_classes_per_label = num_classes_per_label
    def forward(self,x):
        x = x.flatten(1)
        x = self.mlp(x)
        return [head(x) for head in self.classifier_heads]
    
class ClipClassifer(nn.Module):
    def __init__(self,num_classes_per_label:list[int]):
        super().__init__()
        input_dim = 768
        hidden_dim = 512
        self.mlp = nn.Sequential(*[
            nn.Linear(input_dim,hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim,hidden_dim),        
            nn.GELU(),
        ])
        self.classifier_heads = nn.ModuleList([nn.Linear(hidden_dim,num_class) for num_class in num_classes_per_label])
        self.num_classes_per_label = num_classes_per_label
    def forward(self,x):
        x = self.mlp(x)
        return [head(x) for head in self.classifier_heads]
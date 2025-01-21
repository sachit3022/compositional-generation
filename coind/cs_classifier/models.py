
import torch
from torch import nn
from torch.nn import functional as F

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
    


    
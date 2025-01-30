from torchmetrics import Metric
from torchmetrics import Accuracy
from torch import nn
import torch


class MultiLabelAcc(Metric):
    def __init__(self,num_classes_per_label:int,device=None):
        super().__init__()
        self.acc = nn.ModuleList([Accuracy(num_classes=num_classes,task="multiclass") for num_classes in num_classes_per_label])
        if device is not None:
            self.acc = self.acc.to(device)
        self.num_classes_per_label = num_classes_per_label
    def update(self,preds, target):
        for i in range(len(self.num_classes_per_label)):
            self.acc[i].update(preds[i],target[:,i])
    def compute(self):
        return [acc.compute() for acc in self.acc]
    def reset(self):
        for acc in self.acc:
            acc.reset()
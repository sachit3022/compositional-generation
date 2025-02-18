import sys
from pathlib import Path
sys.path.append(str(Path(__file__).absolute().parent.parent))
import os
import logging
import torch
import hydra
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from tqdm import tqdm
from utils import set_seed
from metrics import JSD, CS, Diversity
from score.pipelines import  CondDDIMPipeline
from score.sampling import ANDquery,CFGquery, LaceANDquery,LaceCFGquery
from datasets.sythetic import SytheticData
from datasets.celeba import default_celeba_transform
from cs_classifier.models import MultiLabelClassifier
import torchvision
import csv

if torch.cuda.is_available():
    device = torch.device('cuda')



if __name__ == "__main__":
    
    val_data = SytheticData(root_dir='/research/hal-gaudisac/Diffusion/compositional-generation/samples/vanilla_gender_smile/and_3_24',transform=default_celeba_transform('val',128))
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=128, shuffle=False, num_workers=4)

    cs_classfier =  MultiLabelClassifier(
        base_model=torchvision.models.resnet18(),
        num_classes_per_label=[2,2],
    )
    cs_classfier.load_state_dict(torch.load('/research/hal-gaudisac/Diffusion/compositional-generation/checkpoints/cs_classifier/celeba_gender_smile.ckpt')['state_dict'])
    cs_classfier.to(device)
    cs_classfier.eval()
    correct = [0 for _ in range(2)]
    total = 0
    both = 0
    with torch.no_grad():
        for batch in val_loader:   
            x,y = batch['X'].to(device),batch['label'].to(device)
            logits = cs_classfier(x)
            local_both = torch.tensor([True]).to(device).expand(y.size(0))
            for i in range(2):
                pred = torch.argmax(logits[i],dim=1)
                correct[i] += torch.sum(pred == (y[:,i]+1)//2).item()
                local_both = local_both & (pred == (y[:,i]+1)//2)
            total += y.size(0)
            both += torch.sum(local_both).item()

    print([c/total for c in correct])
    print(both/total)


    #step 2 load the classifier


    
    #report CS score
    #step 3 compare fid against
    #report fid score





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
from torchmetrics.image.fid import FrechetInceptionDistance
from torchvision.models import inception_v3
import torchvision
from torchvision import transforms
import csv
from diffusers import DDPMScheduler


from score.pipelines import  CondDDIMPipeline
from score.sampling import ANDquery,CFGquery, LaceANDquery,LaceCFGquery
from datasets.sythetic import SytheticData
from datasets.celeba import default_celeba_transform, CelebADataset, male_smile_transform, MaleSmileLatent
from cs_classifier.models import MultiLabelClassifier
from models.conditional_unet import ClassConditionalUnet
from models.lace_unet import ComposableUnet

import numpy as np
import itertools
from scipy.special import rel_entr
import torch.nn.functional as F
import torch
import os

from metrics import JSD 


if torch.cuda.is_available():
    device = torch.device('cuda')

class MaleSmileDataset(CelebADataset):
    def filter_data(self,split):
        mask = torch.logical_and(self.attributes_csv.data[:,31] == 1, self.attributes_csv.data[:,20] == 1)
        return mask

class MaleSmileJSD(JSD):
    @torch.no_grad()
    def guidance_evaluator(self,batch,model,scheduler):

        device = self.device
        start,end = self.time_limit
        c_1,c_2 = self.c_1_num_classes,self.c_2_num_classes
        c_1_index,c_2_index = self.c_1_index,self.c_2_index

        x_og,y_og,y_null = batch["X"].to(model.device), batch["label"].to(model.device), batch["label_null"].to(model.device)
        all_possible_joint_labels = torch.tensor(list(itertools.product(range( self.c_1_num_classes),range(self.c_2_num_classes))),device=device,dtype=torch.long)
        all_possible_joint_labels = all_possible_joint_labels*2 -1.0 #we didnot use labels for celebA
        y_all  = y_null[:1].repeat(c_1*c_2+c_1+c_2,1).to(dtype=all_possible_joint_labels.dtype,device=device)
        y_all[:c_1*c_2,c_1_index] = all_possible_joint_labels[:,0]
        y_all[:c_1*c_2,c_2_index] = all_possible_joint_labels[:,1]
        y_all[c_1*c_2:c_1*c_2+c_1,c_1_index] = torch.arange(c_1,device=device)
        y_all[c_1*c_2+c_1:,c_2_index] =torch.arange(c_2,device=device)

        y_true = y_all.repeat(self.num_of_timesteps,1)
        
        
        og_timesteps = torch.linspace(start, end, self.num_of_timesteps,device=device).long()
        timesteps = og_timesteps.repeat_interleave(y_all.size(0))

        accuracy = [0,0]
        js_divergence = []
        x_og,y_og = batch["X"].to(model.device), batch["label"].to(model.device)
        for i in range(x_og.size(0)):
            x0 = x_og[i:i+1]
            y = y_og[i:i+1]
            
            noise = torch.randn_like(x0)
            xt = scheduler.add_noise(x0, noise, og_timesteps)
            xt = xt.repeat_interleave(y_all.size(0),dim=0)
            with torch.no_grad():
                noise_pred = model(xt, timesteps,y_true)
            n_chunks = [n.mean(dim=(1,2,3)) for n in F.mse_loss(noise_pred, noise, reduction='none').chunk(self.num_of_timesteps,dim=0)]
            l = sum([(n - n.mean())/n.std() for n in n_chunks]).view(y_all.size(0),-1).mean(dim=1).detach().cpu()
            joint =F.softmax(-1*l[:c_1*c_2],dim=0).numpy()
            indp_0 = F.softmax( -1*l[c_1*c_2:c_1*c_2+c_1],dim=0)
            indp_1 = F.softmax(-1*l[c_1*c_2+c_1:],dim=0)
            mutual = ((indp_0.reshape(-1,1) @ indp_1.reshape(1,-1))).numpy()
            js_divergence_m = 0.5*(sum(rel_entr(mutual.reshape(-1),joint.reshape(-1)))+ sum(rel_entr(joint.reshape(-1),mutual.reshape(-1))))
            js_divergence.append(js_divergence_m)
            y_est = [np.argmax(indp_0) == y[0,c_1_index].cpu().numpy(), np.argmax(indp_1)== y[0,c_2_index].cpu().numpy()]
            accuracy = [a+b for a,b in zip(accuracy,y_est)]
        return sum(js_divergence)/x_og.size(0),[x*1.0/len(x_og) for x in accuracy]





if __name__ == "__main__":

    ################# JSD  #################
    # checkpoint_path = '/research/hal-gaudisac/Diffusion/compositional-generation/checkpoints/celeba/celeba_partial_coind.ckpt'
    # model_config = {
    #     '_target_': ClassConditionalUnet,
    #     'num_class_per_label': [2,2],
    #     'sample_size': 16,
    #     'in_channels': 16,
    #     'out_channels': 16, 
    #     'center_input_sample': False,
    #     'time_embedding_type': 'positional',
    # }
    # scheduler_config = {
    #     '_target_': DDPMScheduler,
    #     'num_train_timesteps': 1000,
    #     "clip_sample": True,
    #     "prediction_type": "epsilon",
    #     'beta_schedule': 'squaredcos_cap_v2',
    # }
    # model = hydra.utils.instantiate(model_config)
    # scheduler = hydra.utils.instantiate(scheduler_config)
    # model_state_dict = {k.replace('model.',''): v for k, v in torch.load(checkpoint_path,weights_only=False,map_location=device)['state_dict'].items() if k.startswith('model.')}
    # model.load_state_dict(model_state_dict, strict=False)
    # model.to(device)

    # celeba_dir=   "/research/hal-datastore/datasets/original/"
    # latent_dir = "/research/hal-gaudisac/Diffusion/compositional-generation/data/celeba_male_smile/vae_val_features"
    # val_data = MaleSmileLatent( celeba_dir,latent_dir,split='val')
    # val_loader = torch.utils.data.DataLoader(val_data, batch_size=32, shuffle=False, num_workers=4)

    
    # jsd = MaleSmileJSD(model.num_classes_per_label).to(model.device)
    # count = 0
    # for epoch,batch in enumerate(tqdm(val_loader)):
    #     jsd.update(batch, model,scheduler)
    #     count += batch["X"].size(0)
    #     if count >=1000:
    #         break 
    # print(jsd.compute())

    ################# CS AND FID #################

    print("before CS and FID please run generate_male_smile_samples.py")
    
    main_folder = "/research/hal-gaudisac/Diffusion/compositional-generation/samples/lace_gender_smile/"
    
    cs_classfier =  MultiLabelClassifier(
            base_model=torchvision.models.resnet18(),
            num_classes_per_label=[2,2],
        )
    cs_classfier.load_state_dict(torch.load('/research/hal-gaudisac/Diffusion/compositional-generation/checkpoints/cs_classifier/celeba_gender_smile.ckpt')['state_dict'])
    cs_classfier.to(device)
    cs_classfier.eval()

    #get all the folders
    for folder in os.listdir(main_folder):

        folder = os.path.join(main_folder,folder)

        ### CS Score
        val_data = SytheticData(root_dir=folder,transform=default_celeba_transform('val',128))
        val_loader = torch.utils.data.DataLoader(val_data, batch_size=128, shuffle=False, num_workers=4)
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
                    correct[i] += torch.sum(pred == 1.0).item()
                    local_both = local_both & (pred == 1.0)
                total += y.size(0)
                both += torch.sum(local_both).item()
            

        print(folder)
        print([c/total for c in correct])
        print(both/total)
    
        # Initialize FID metric
        fid_transform= transforms.Compose([
                transforms.Resize((128, 128)),
                transforms.PILToTensor(),
            ])
        train_data = MaleSmileDataset(root='/research/hal-datastore/datasets/original/',split=None,size=128,transforms=fid_transform,target_transform=male_smile_transform())
        train_loader = torch.utils.data.DataLoader(train_data, batch_size=128, shuffle=False, num_workers=4)

        ### FID Score
        fid = FrechetInceptionDistance(feature=2048).to(device)
        for batch in train_loader:
            fid.update(batch['X'].to(device), real=True)

        val_data = SytheticData(root_dir=folder,transform=fid_transform)
        val_loader = torch.utils.data.DataLoader(val_data, batch_size=128, shuffle=False, num_workers=4)
        for batch in val_loader:
            fid.update(batch['X'].to(device), real=False)

        print(f"FID: {fid.compute()}")






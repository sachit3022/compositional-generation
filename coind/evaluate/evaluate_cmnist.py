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
from score.pipelines import  CondDDIMPipeline, 
from score.sampling import ANDquery,CFGquery, ModelWrapper,LaceModelWrapper, LaceANDquery
from models.lace_unet import ComposableUnet
import csv

if torch.cuda.is_available():
    device = torch.device('cuda')

def prepare_and_query(y,null_token,guidance_scale):
    B,Q = y.size()
    query = null_token.repeat(Q,1).to(dtype=y.dtype,device=y.device)
    query = query.reshape(B,Q,Q)
    for i in range(Q):
        query[:,i,i] = y[:,i]
    guidance_scale = [guidance_scale]*Q
    return query,guidance_scale

def digit_not_query(y,null_token,guidance_scale):
    B,Q = y.size()
    assert Q == 2
    query = null_token.repeat(3,1).to(dtype=y.dtype,device=y.device)
    query = query.reshape(B,3,2)
    query[:,0,1] = y[:,1]
    query[:,1,0] = y[:,1]
    query[:,2,0] = torch.clamp(y[:,1]-1, min=0, max=9)
    guidance_scale = guidance_scale*3
    guidance_scale = torch.tensor([2*guidance_scale,-1*guidance_scale,-1*guidance_scale]).to(query.device)
    return query,guidance_scale

def color_not_query(y,null_token,guidance_scale):
    B,Q = y.size()
    assert Q == 2
    query = null_token.repeat(3,1).to(dtype=y.dtype,device=y.device)
    query = query.reshape(B,3,2)
    query[:,0,0] = y[:,0]
    query[:,1,1] = y[:,0]
    query[:,2,1] = torch.clamp(y[:,0]+1, min=0, max=9)
    guidance_scale = guidance_scale*2
    guidance_scale = torch.tensor([2*guidance_scale,-1*guidance_scale,-1*guidance_scale]).to(query.device)
    return query,guidance_scale

class DigitNotquery(ModelWrapper):
    def prepare_query(self,y,null_token,guidance_scale):
        return digit_not_query(y,null_token,guidance_scale)
class ColorNotquery(ModelWrapper):
    def prepare_query(self,y,null_token,guidance_scale):
        return color_not_query(y,null_token,guidance_scale)


class LaceDigitNotquery(LaceModelWrapper):
    def prepare_query(self,y,null_token,guidance_scale):
        return digit_not_query(y,null_token,guidance_scale)
class LaceColorNotquery(LaceModelWrapper):
    def prepare_query(self,y,null_token,guidance_scale):
        return color_not_query(y,null_token,guidance_scale)


def log_metrics(evaluation,jsd,diversity, prefix,epoch,output_dir):
    # format and log metrics into nice dict and save to csv
    csv_dict = {"epoch": epoch}
    for cs_name,cs_dict in evaluation.items():
        for metric_subname, metric in cs_dict["metric"].compute().items():
            csv_dict[f'{prefix}/{cs_name}/{metric_subname}'] = metric.item()
    
    if jsd.count > 0:
        for metric_subname, metric in jsd.compute().items():
            csv_dict[f'{prefix}/jsd/{metric_subname}'] = metric.item()
    
    
    for i,div_metric in enumerate(diversity):
        for metric_subname, metric in div_metric["metric"].compute().items():
            csv_dict[f'{prefix}/{i}_diversity/{metric_subname}'] = metric

    #How to log to a csv file the keys might be different
    filename = os.path.join(output_dir, f'{prefix}_metrics.csv')
    file_exists = os.path.isfile(filename)
    with open(filename, 'a') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=csv_dict.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(csv_dict)



def reset_metrics(evaluation,jsd,diversity):
    for cs_name,cs_dict in evaluation.items():
        cs_dict["metric"].reset()
    jsd.reset()
    for div_metric in diversity:
        div_metric["metric"].reset()


@hydra.main(config_path='../../configs',version_base='1.2',config_name='cmnist_inference')
def main(cfg):
    
    ########## Hyperparameters and settings ##########
    set_seed(cfg.seed)
    output_dir = HydraConfig.get().runtime.output_dir
    logger = logging.getLogger(__name__)
    ########## Dataset ##########
    
    train_dataset = instantiate(cfg.dataset.train_dataset)
    val_dataset = instantiate(cfg.dataset.val_dataset)
   
    train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset,**cfg.dataset.train_dataloader)
    val_dataloader = torch.utils.data.DataLoader(dataset=val_dataset,**cfg.dataset.val_dataloader)

    ########## Load diffusion model #############

    model = instantiate(cfg.model).to(device)
    scheduler = instantiate(cfg.noise_scheduler)
    checkpoint_path = cfg.checkpoint_path
    model_state_dict = {k.replace('model.',''): v for k, v in torch.load(checkpoint_path)['state_dict'].items() if k.startswith('model.')}
    model.load_state_dict(model_state_dict, strict=False)
    model.eval()

    # ########## Metrics #############
    ### hyperparameters ##
    num_inference_steps = 100
    guidance = 7.5
    max_samples = 10_000

    classifier = instantiate(cfg.classifier)
    classifier.load_state_dict(torch.load(cfg.classifer_checkpoint)["state_dict"])

    #evaluate for multiple queries
    if isinstance(model, ComposableUnet):
        cs_evaluations = {"val":{
                "and": {"sampler":CondDDIMPipeline(unet=LaceANDquery(model), scheduler=scheduler),"query":prepare_and_query,"metric":CS(classifier = classifier).to(device)},
                "digit_not": {"sampler":CondDDIMPipeline(unet=LaceDigitNotquery(model), scheduler=scheduler),"query":digit_not_query,"metric":CS(classifier = classifier).to(device)},
                "color_not":{"sampler":CondDDIMPipeline(unet=LaceColorNotquery(model), scheduler=scheduler),"query":color_not_query,"metric":CS(classifier = classifier).to(device)}
            },
            "train":{
                "and": {"sampler":CondDDIMPipeline(unet=LaceANDquery(model), scheduler=scheduler),"query":prepare_and_query,"metric":CS(classifier = classifier).to(device)},            }
    }
    else:
        cs_evaluations = {"val":{
                    "and": {"sampler":CondDDIMPipeline(unet=ANDquery(model), scheduler=scheduler),"query":prepare_and_query,"metric":CS(classifier = classifier).to(device)},
                    "joint": {"sampler":CondDDIMPipeline(unet=CFGquery(model), scheduler=scheduler),"query":prepare_and_query,"metric":CS(classifier = classifier).to(device)},
                    "digit_not": {"sampler":CondDDIMPipeline(unet=DigitNotquery(model), scheduler=scheduler),"query":digit_not_query,"metric":CS(classifier = classifier).to(device)},
                    "color_not":{"sampler":CondDDIMPipeline(unet=ColorNotquery(model), scheduler=scheduler),"query":color_not_query,"metric":CS(classifier = classifier).to(device)}
                },
                "train":{
                    "and": {"sampler":CondDDIMPipeline(unet=ANDquery(model), scheduler=scheduler),"query":prepare_and_query,"metric":CS(classifier = classifier).to(device)},
                    "joint": {"sampler":CondDDIMPipeline(unet=CFGquery(model), scheduler=scheduler),"query":prepare_and_query,"metric":CS(classifier = classifier).to(device)},
                }
        }
    if not isinstance(model, ComposableUnet):
        diversity = [ {"sampler":CondDDIMPipeline(unet=CFGquery(model), scheduler=scheduler),"metric":Diversity(classifier = classifier).to(device)},
                        {"sampler":CondDDIMPipeline(unet=CFGquery(model), scheduler=scheduler),"metric":Diversity(classifier = classifier).to(device)}]
    else:
        diversity = [ {"sampler":CondDDIMPipeline(unet=LaceANDquery(model), scheduler=scheduler),"metric":Diversity(classifier = classifier).to(device)},
                        {"sampler":CondDDIMPipeline(unet=LaceANDquery(model), scheduler=scheduler),"metric":Diversity(classifier = classifier).to(device)}]

    pipleline_kargs = {'num_inference_steps':num_inference_steps,
                        'return_dict':True,
                        'use_clipped_model_output':True,
                        'guidance_scale':guidance}

    for pbar in [ tqdm(val_dataloader, desc="val"),tqdm(train_dataloader, desc="train")]:
        evaluation = cs_evaluations[pbar.desc]
        jsd = JSD(model.num_classes_per_label).to(model.device)
        for epoch,batch in enumerate(pbar):
            x,y,null_token  = batch["X"].to(device), batch["label"].to(device), batch["label_null"].to(device)
            ########### JSD #############
            if not isinstance(model, ComposableUnet):
                jsd.update(batch, model,scheduler)
            ########### CS #############
            for cs_name,cs_dict in evaluation.items():
                generated_images = cs_dict["sampler"](batch_size= x.size(0),query = y,null_token=null_token,**pipleline_kargs)[0]
                query,guidance_scale = cs_dict["query"](y,null_token,guidance)
                cs_dict["metric"].update(generated_images,query,null_token,guidance_scale)
            ######### Diversity ##########
            if pbar.desc == "val":
                for i in range(2):
                    query = y.clone()
                    query[:,i] = null_token[:,i]
                    generated_images = diversity[i]['sampler'](batch_size= x.size(0),query = query,null_token=null_token,**pipleline_kargs)[0]
                    diversity[i]['metric'].update(generated_images,query,null_token)
            else:
                diversity = []

            log_metrics(evaluation,jsd,diversity, pbar.desc,epoch,output_dir)
            if (epoch+1)*batch["X"].size(0) > max_samples:
                break
        reset_metrics(evaluation,jsd,diversity)

if __name__ == "__main__":
    main()
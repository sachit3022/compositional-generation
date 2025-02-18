#typically for load model from the config but its okay we will define the config as a dict
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).absolute().parent.parent))
import os
import torch
import math
import hydra
from diffusers import DDPMScheduler,AutoencoderKL

from models.conditional_unet import ClassConditionalUnet
from score.pipelines import CondDDIMPipeline
from score.sampling import ANDquery,CFGquery, LaceANDquery,LaceCFGquery
from utils import set_seed
from argparse import ArgumentParser

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp

import torchvision.utils as vutils
import json
from filelock import FileLock

import os
import numpy as np


os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '29500'

        
def save_images(query,null_token,generated_images,rank,save_folder,iteration=0):
    metadata_list = []
   
    for image_index, (image, q,nt) in enumerate(zip(generated_images, query,null_token)):
        save_index = iteration+image_index
        filename = f"image_{rank}_{save_index:06d}.png"  # Creates filenames like image_000.png, image_001.png, etc.
        save_path = os.path.join(save_folder, "gen", filename)
        vutils.save_image(image, save_path, normalize=True)
        metadata = {'filename':filename, 'query': q.tolist(), 'null_token': nt.tolist()}
        metadata_list.append(metadata)
    
    
    lock_file = os.path.join(save_folder, 'metadata.lock')
    metadata_file = os.path.join(save_folder, 'metadata.json')

    with FileLock(lock_file):
        with open(metadata_file, 'a') as f:
            for metadata in metadata_list:
                json.dump(metadata, f)
                f.write('\n')


def setup(rank, world_size):
    dist.init_process_group("nccl", rank=rank, world_size=world_size, init_method="env://")


def get_image_generation_pipeline(checkpoint_path,rank,composition):

    model_config = {
        '_target_': ClassConditionalUnet,
        'num_class_per_label': [2,2],
        'sample_size': 16,
        'in_channels': 16,
        'out_channels': 16, 
        'center_input_sample': False,
        'time_embedding_type': 'positional',
    }

    scheduler_config = {
        '_target_': DDPMScheduler,
        'num_train_timesteps': 1000,
        "clip_sample": True,
        "prediction_type": "epsilon",
        'beta_schedule': 'squaredcos_cap_v2',
    }

    model = hydra.utils.instantiate(model_config)
    scheduler = hydra.utils.instantiate(scheduler_config)

    #load model
    model_state_dict = {k.replace('model.',''): v for k, v in torch.load(checkpoint_path,weights_only=False,map_location=f"cuda:{rank}")['state_dict'].items() if k.startswith('model.')}
    model.load_state_dict(model_state_dict, strict=False)
    model.to(f"cuda:{rank}")
    model.eval()


    vae = AutoencoderKL.from_pretrained('black-forest-labs/FLUX.1-schnell', subfolder='vae', cache_dir='checkpoints', device_map=rank)
    vae.eval()
    and_model = composition(model)
    pipeline = CondDDIMPipeline(unet=and_model, scheduler=scheduler, vae=vae)
    return pipeline

def generate_images(rank, world_size, checkpoint, num_images_per_gpu,filepath,composition,guidance_scale,query,null_token):
    setup(rank, world_size)
    torch.cuda.set_device(rank)
    batch_size = 32
    pipeline = get_image_generation_pipeline(checkpoint,rank,composition=composition)
    set_seed(rank)
    num_inference_steps = 250
    os.makedirs(f"{filepath}/gen",exist_ok=True)
    query = query.repeat(batch_size,1).to(rank)
    null_token = null_token.repeat(batch_size,1).to(rank)
    for i in range(num_images_per_gpu//batch_size):
        with torch.no_grad():
            generated_images = pipeline(batch_size=query.size(0), 
                            num_inference_steps= num_inference_steps,
                            return_dict= True,
                            use_clipped_model_output = True,
                            query = query,
                            guidance_scale=guidance_scale,
                            null_token=null_token)[0].cpu().detach()*0.5+0.5 #denormalize
            save_images(query,null_token,generated_images,rank,filepath,i*batch_size)


if __name__ == '__main__':
    """
    make this to run on distributed process to run accross multiple gpus.
    """
    checkpoints = {
        "vanilla":"/research/hal-gaudisac/Diffusion/compositional-generation/checkpoints/celeba/celeba_partial_diffusion.ckpt",
        "coind":'/research/hal-gaudisac/Diffusion/compositional-generation/checkpoints/celeba/celeba_partial_coind.ckpt',
    }
    args = ArgumentParser()
    args.add_argument('--num_of_images',type=int,default=10_000)
    args.add_argument('--checkpoint',type=str,default=checkpoints['coind'])
    args = args.parse_args()
    
    world_size = torch.cuda.device_count()
    num_images_per_gpu = args.num_of_images // world_size
    exp = "coind_gender_smile" 

    settings_to_evaluate = [
    {
        "filepath": f"samples/{exp}/and_9_9",
        "guidance_scale":[9.0,9.0],
        "query":torch.tensor([[1.0,1.0]]), 
        "composition": ANDquery
    }, 
    {
        "filepath": f"samples/{exp}/and_9_18",
        "guidance_scale":[9.0,18.0],
        "query":torch.tensor([[1.0,1.0]]), 
        "composition": ANDquery
    },
    {
        "filepath": f"samples/{exp}/and_9_27",
        "guidance_scale":[9.0,27.0],
        "query":torch.tensor([[1.0,1.0]]), 
        "composition": ANDquery

    },
    {
        "filepath": f"samples/{exp}/and_3_24",
        "guidance_scale":[3.0,24.0],
        "query":torch.tensor([[1.0,1.0]]), 
        "composition": ANDquery
    },
    {
        "filepath": f"samples/{exp}/exp16/cfg_9",
        "guidance_scale":9.0,
        "query":torch.tensor([[1.0,1.0]]), 
        "composition": CFGquery
    },
    {
        "filepath": f"samples/{exp}/neg_smile_9",
        "guidance_scale":[9.0,-9.0],
        "query":torch.tensor([[1.0,-1.0]]), 
        "composition": ANDquery
    },
    {
        "filepath": f"samples/{exp}/neg_gender_9",
        "guidance_scale":[-9.0,9.0],
        "query":torch.tensor([[-1.0,1.0]]), 
        "composition": ANDquery
    }
    ]

    for setting in settings_to_evaluate:
        filepath = setting['filepath']
        guidance_scale = setting['guidance_scale']
        query = setting['query']
        composition = setting['composition']
        null_token = torch.zeros_like(query)
        mp.spawn(generate_images, args=(world_size,args.checkpoint,num_images_per_gpu,filepath,composition,guidance_scale,query,null_token ), nprocs=world_size)    
    dist.destroy_process_group()


    
    


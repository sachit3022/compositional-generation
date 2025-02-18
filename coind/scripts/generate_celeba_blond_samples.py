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
from score.sampling import ANDquery,CFGquery
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
def save_latent(query,null_token,generated_images,rank,save_folder,iteration=0):
    metadata_list = []
    for image_index, (image, q,nt) in enumerate(zip(generated_images, query,null_token)):
        save_index = iteration+image_index
        filename = f"image_{rank}_{save_index:06d}.npy" 
        save_path = os.path.join(save_folder, "gen", filename)
        np.save(save_path, image)
        metadata = {'filename':filename, 'query': q.tolist(), 'null_token': nt.tolist()}
        metadata_list.append(metadata)
        
    lock_file = os.path.join(save_folder, 'metadata.lock')
    metadata_file = os.path.join(save_folder, 'metadata.json')

    with FileLock(lock_file):
        with open(metadata_file, 'a') as f:
            for metadata in metadata_list:
                json.dump(metadata, f)
                f.write('\n')
        

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


def get_image_generation_pipeline(checkpoint_path,latent=False):

    model_config = {
        '_target_': ClassConditionalUnet,
        'num_class_per_label': [2,2],
        'interaction': 'sum',
        'sample_size': 8,
        'in_channels': 16,
        'out_channels': 16, 
        'center_input_sample': False,
        'time_embedding_type': 'positional'
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
    model_state_dict = {k.replace('model.',''): v for k, v in torch.load(checkpoint_path,weights_only=False)['state_dict'].items() if k.startswith('model.')}
    model.load_state_dict(model_state_dict, strict=False)

    #load vae
    vae = None
    if not latent:
        vae = AutoencoderKL.from_pretrained('stabilityai/stable-diffusion-3-medium-diffusers', subfolder='vae', cache_dir='checkpoints')        
        vae.eval()

    model.eval()
    
    and_model = ANDquery(model)
    pipeline = CondDDIMPipeline(unet=and_model, scheduler=scheduler, vae=vae)
    return pipeline

def generate_images(rank, world_size, checkpoint, num_images_per_gpu,latent):
    setup(rank, world_size)
    torch.cuda.set_device(rank)
    
    pipeline = get_image_generation_pipeline(checkpoint,latent).to(rank)
    if not latent:
        pipeline.vae.to(rank)

    set_seed(rank)

    batch_size = 64
    guidance_scale = [7.5,3.5]
    num_inference_steps = 250
    filepath = "samples/exp14"
    os.makedirs(f"{filepath}/gen",exist_ok=True)
    for i in range(num_images_per_gpu//batch_size):

        with torch.no_grad():

            query = torch.tensor([[0,1]]).to(rank).repeat(batch_size,1) #torch.randint(0,2,(batch_size,2)).to(rank)
            null_token = torch.ones_like(query).to(rank)*2
            
            generated_images = pipeline(batch_size=query.size(0), 
                            num_inference_steps= num_inference_steps,
                            return_dict= True,
                            use_clipped_model_output = True,
                            query = query,
                            guidance_scale=guidance_scale,
                            null_token=null_token)[0].cpu().detach()
            save_func = save_latent if latent else save_images
            save_func(query,null_token,generated_images,rank,filepath,i*batch_size)

    dist.destroy_process_group()


if __name__ == '__main__':
    """
    make this to run on distributed process to run accross multiple gpus.
    """
    checkpoints = {
        "vanilla":"/research/hal-gaudisac/Diffusion/compositional-generation/outputs/2025-01-29/12-52-01/tensorboard/version_0/checkpoints/s.ckpt",
        "coind":'/research/hal-gaudisac/Diffusion/compositional-generation/outputs/2025-01-29/13-48-54/tensorboard/version_0/checkpoints/s.ckpt',
        "full":"/research/hal-gaudisac/Diffusion/compositional-generation/outputs/2025-01-29/12-08-26/tensorboard/version_0/checkpoints/s.ckpt"
    }

    args = ArgumentParser()
    args.add_argument('--num_of_images',type=int,default=20_000)
    args.add_argument('--checkpoint',type=str,default=checkpoints['vanilla'])
    args.add_argument('--latent',type=bool,default=False)


    args = args.parse_args()

    world_size = torch.cuda.device_count()
    num_images_per_gpu = args.num_of_images // world_size
    
    mp.spawn(generate_images, args=(world_size,args.checkpoint,num_images_per_gpu,args.latent), nprocs=world_size)

    



    
    


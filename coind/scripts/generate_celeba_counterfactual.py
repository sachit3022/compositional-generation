#typically for load model from the config but its okay we will define the config as a dict
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).absolute().parent.parent))
import os
import torch
import math
import torch.multiprocessing as mp

import hydra
from diffusers import DDPMScheduler,AutoencoderKL
from diffusers.pipelines.pipeline_utils import ImagePipelineOutput

from models.conditional_unet import ClassConditionalUnet
from score.pipelines import CondDDIMPipeline
from score.sampling import ANDquery,CFGquery
from datasets.celeba import BlondFemaleDataset,default_celeba_transform
from utils import set_seed
from argparse import ArgumentParser

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


import torchvision.utils as vutils
from torch.utils.data import DataLoader
import json
from filelock import FileLock

import os
import numpy as np

from typing import Optional, Union, List


os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '29503'
        



class CounterfactualDDIMPipeline(CondDDIMPipeline):
    """
    A PyTorch Lightning module that implements the DDIM pipeline for image data. Taken from the original DDIM implementation."""
    @torch.no_grad()
    def __call__(
        self,
        batch_size: int = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        eta: float = 0.0,
        num_inference_steps: int = 50,
        use_clipped_model_output: Optional[bool] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        query: Optional[torch.Tensor] = None,
        guidance_scale: Union[float,int,torch.Tensor] = 0.0,
        null_token: Optional[torch.Tensor] = None,
        image: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        if isinstance(self.unet.config.sample_size, int):
            image_shape = (
                batch_size,
                self.unet.config.in_channels,
                self.unet.config.sample_size,
                self.unet.config.sample_size,
            )
        elif self.unet.config.in_channels ==0:
            image_shape = (batch_size, *self.unet.config.sample_size)
        else:
            image_shape = (batch_size, self.unet.config.in_channels, *self.unet.config.sample_size)

        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        if image is None:
            image = randn_tensor(image_shape, generator=generator, device=self.unet.device, dtype=self.unet.dtype)
    
        self.scheduler.set_timesteps(num_inference_steps)
        if 'noise_percentage' in kwargs:
            timesteps = torch.ones(image.size(0), device=self.unet.device, dtype=torch.long)*self.scheduler.timesteps[int(len(self.scheduler.timesteps)*kwargs['noise_percentage'])]
            noise = torch.randn_like(image)
            image = self.scheduler.add_noise(image,noise ,timesteps)
            self.scheduler.timesteps =self.scheduler.timesteps[int(len(self.scheduler.timesteps)*kwargs['noise_percentage'])+1:]

        if query is not None:
            image = image.to(device=query.device)

        for t in self.progress_bar(self.scheduler.timesteps):
            
            # 2. predict previous mean of image x_t-1 and add variance depending on eta
            # eta corresponds to η in paper and should be between [0, 1]
            # do x_t -> x_t-1
            #train_timesteps
            model_output = self.unet(image, t,query, guidance_scale=guidance_scale, null_token=null_token)
            image = self.scheduler.step(
                model_output, t, image, eta=eta, use_clipped_model_output=use_clipped_model_output, generator=generator
            ).prev_sample

        if self.vae is not None:
            image = self.vae.decode(image/self.vae.config.scaling_factor)[0]
        if not return_dict:
            return (image,)
        return ImagePipelineOutput(images=image)

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


def get_image_generation_pipeline(checkpoint_path,rank):

    model_config = {
        '_target_': ClassConditionalUnet,
        'num_class_per_label': [2,2],
        'interaction': 'sum',
        'sample_size': 64,
        'in_channels': 3,
        'out_channels': 3, 
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
    model_state_dict = {k.replace('model.',''): v for k, v in torch.load(checkpoint_path,weights_only=False,map_location=f"cuda:{rank}")['state_dict'].items() if k.startswith('model.')}
    model.load_state_dict(model_state_dict, strict=False)

    model.eval()
    model.to(f"cuda:{rank}")
    
    and_model = ANDquery(model)
    pipeline = CounterfactualDDIMPipeline(unet=and_model, scheduler=scheduler)
    return pipeline

def generate_counterfactual_images(rank, world_size, checkpoint, num_images_per_gpu, og_dataset):
    setup(rank, world_size)
    torch.cuda.set_device(rank)

    pipeline = get_image_generation_pipeline(checkpoint,rank)


    

    set_seed(rank)
    guidance_scale = [3.5, 7.5] #[7.5,7.5]
    batch_size = 12
    num_inference_steps = 100
    noise_percentage = 0.2
    filepath = "samples/exp35"
    i = 0


    og_dataloader = DataLoader(og_dataset, batch_size=batch_size, shuffle=True, num_workers=4, persistent_workers = True)
    
    os.makedirs(f"{filepath}/gen",exist_ok=True)
    for batch in og_dataloader:
        with torch.no_grad():
            image,query,null_token= batch['X'].to(rank),batch['label'].to(rank), batch['label_null'].to(rank)
             
            #randomly make 1/3 of labels as [0,1]
            mask = torch.rand(query.size(0))<0.33
            # add these masks at the end of image
            image = torch.cat([image,image[mask]],dim=0)
            query = torch.cat([query,torch.tensor([0,1],device=query.device).repeat(mask.sum(),1)],dim=0)
            null_token = torch.cat([null_token,null_token[mask]],dim=0)

            generated_images = pipeline(batch_size=query.size(0), 
                            num_inference_steps= num_inference_steps,
                            return_dict= True,
                            use_clipped_model_output = True,
                            query = query,
                            image = image,
                            guidance_scale=guidance_scale,
                            noise_percentage = noise_percentage,
                            null_token=null_token)[0].cpu().detach()*0.5+0.5 #denormalize
            save_images(query,null_token,generated_images,rank,filepath,i*batch_size)
        num_images_per_gpu -= query.size(0)
        i += 1
        if num_images_per_gpu <= 0:
            break

    dist.destroy_process_group()

def gender_blond_transform(x):
    return (x[[20,9]]+1)//2


if __name__ == '__main__':
    """
    make this to run on distributed process to run accross multiple gpus.
    """
    checkpoints = {    
         "vanilla": "/research/hal-gaudisac/Diffusion/compositional-generation/outputs/2025-03-04/23-59-59/tensorboard/version_0/checkpoints/epoch=114-step=500000.ckpt",
        "coind": "/research/hal-gaudisac/Diffusion/compositional-generation/outputs/2025-03-04/23-56-46/tensorboard/version_0/checkpoints/epoch=114-step=500000.ckpt"
    }

    args = ArgumentParser()
    args.add_argument('--num_of_images',type=int,default=20_000)
    args.add_argument('--checkpoint',type=str,default=checkpoints['coind'])


    args = args.parse_args()

    world_size = torch.cuda.device_count()
    num_images_per_gpu = args.num_of_images // world_size
    og_dataset = BlondFemaleDataset('/research/hal-datastore/datasets/original/', split='train',transforms=default_celeba_transform('val'),target_transform=gender_blond_transform)
    mp.spawn(generate_counterfactual_images, args=(world_size,args.checkpoint,num_images_per_gpu,og_dataset), nprocs=world_size)

    



    
    


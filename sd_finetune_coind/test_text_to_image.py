import torch
from diffusers import StableDiffusionPipeline, UNet2DConditionModel
from composable_diffusion.pipeline_composable_stable_diffusion import ComposableStableDiffusionPipeline
import torchvision
from PIL import Image
import random
import numpy as np
from datasets.celeba import image_grid
import json

from diffusers import StableDiffusionPipeline
import torch
#argparse
import argparse
import os
import torch
import torch
from torch import autocast
import os

has_cuda = torch.cuda.is_available()
device = torch.device('cpu' if not has_cuda else 'cuda')




#intial_model ="runwayml/stable-diffusion-v1-5"
intial_model = "stabilityai/stable-diffusion-2-1"
CACHE_DIR="/research/hal-gaudisac/Diffusion/controllable-generation/celeba"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WORKING_DIR = os.path.join(BASE_DIR, "..")

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, default=None, help="prompt for the model", required=True)
    parser.add_argument("--model", type=str, default='', help="one of celeba_sd15, celeba_sd15_lora, celeba_sd15_coind, celeba_sd15_coind_lora", required=False)
    parser.add_argument("--weights", type=str, default=None, help="path to the model weight", required=False)
    parser.add_argument("--seed", type=int, default=42, help="seed for the model", required=False)
    parser.add_argument("--num_of_images", type=int, default=10, help="number of images to generate", required=False)
    return parser.parse_args()


if __name__ == "__main__":
    """
    Examples:
        CUDA_VISIBLE_DEVICES=0 /research/hal-gaudisac/miniconda3/bin/python test_text_to_image.py --prompt='Photo of a smiling female celebrity' --model='celeba_sd15_coind_lora' --seed=42
    """
    N = 30000
    num_inference_steps = 100
    num_images_per_prompt = 10 #max that can fit gpu memory
    args = get_args()

    
    prompt = args.prompt
    model = args.model
    compose = '|' in prompt
    seed = args.seed
    num_of_images=  args.num_of_images
    set_seed(seed)

    print(f"Prompt: {prompt}")
    print(f"Model: {model}/checkpoint-{N}")
    print(f"Compose: {compose}")
    # initialize stable diffusion model
    if compose:
        Pipeline = ComposableStableDiffusionPipeline
    else:
        Pipeline = StableDiffusionPipeline

    kwargs = {}
    if compose and args.weights:
        kwargs['weights'] = args.weights
        kwargs["negative_prompt"] = ['Photo of a celebrity']
    else:
        kwargs['guidance_scale']=7.5
        kwargs["negative_prompt"] = 'Photo of a celebrity' #a photo of a bird #'Photo of a celebrity'

    if model=='':
        pipe = Pipeline.from_pretrained(intial_model, torch_dtype=torch.float16, cache_dir=CACHE_DIR)
    elif 'lora' in model:
        model_path = f"{BASE_DIR}/../outputs/{model}/checkpoint-{N}"
        pipe = Pipeline.from_pretrained(intial_model, torch_dtype=torch.float16, cache_dir=CACHE_DIR)
        pipe.unet.load_attn_procs(model_path)
    else:
        model_path = f"{BASE_DIR}/../outputs/{model}/checkpoint-{N}/unet"
        unet = UNet2DConditionModel.from_pretrained(model_path, torch_dtype=torch.float16)
        pipe = Pipeline.from_pretrained(intial_model, torch_dtype=torch.float16, cache_dir=CACHE_DIR,unet=unet)
    
    pipe.to(device)
    pipe.safety_checker = None

    #want to save 100 images in the model folder
    save_dir = f"{BASE_DIR}/../images/{model}_sd2_neg"
    print(f"Saving images to {save_dir}")

    if not os.path.exists(f"{save_dir}"):
        os.makedirs(f"{save_dir}")

    with open(f"{save_dir}/prompt.txt", "w") as f:
        config = {
            "prompt":prompt,
            "seed":seed,
            'compose':compose,
            'model':f"{model}/checkpoint-{N}",
            'save_dir':save_dir,
        }
        json.dump(config, f)
    
    image_count = 0
    #if multi gpu then parallelize generation accross gpus

    for i in range(num_of_images//num_images_per_prompt):
        with torch.autocast("cuda"):
            images= pipe(prompt=prompt,num_images_per_prompt=num_images_per_prompt,num_inference_steps=num_inference_steps,**kwargs).images
            for i in range(num_images_per_prompt):
                #convert to 178×218
                img = images[i].resize((178,218))
                img.save(f"{save_dir}/{image_count}.jpg")
                image_count+=1
    
        

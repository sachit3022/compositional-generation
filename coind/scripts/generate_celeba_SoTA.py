
from diffusers import UNet2DModel, DDIMScheduler, VQModel
import torch
import PIL.Image
import numpy as np
import tqdm


def generate_celeba_from_sota_model(image,noise_percentage=0.7,num_inference_steps=100,device='cuda'):
    seed = 42
    # load all models
    unet = UNet2DModel.from_pretrained("CompVis/ldm-celebahq-256", subfolder="unet",cache_dir="/research/hal-gaudisac/Diffusion/compositional-generation/checkpoints",allow_pickle=False)
    vqvae = VQModel.from_pretrained("CompVis/ldm-celebahq-256", subfolder="vqvae",cache_dir="/research/hal-gaudisac/Diffusion/compositional-generation/checkpoints",allow_pickle=False)
    scheduler = DDIMScheduler.from_config("CompVis/ldm-celebahq-256", subfolder="scheduler",cache_dir="/research/hal-gaudisac/Diffusion/compositional-generation/checkpoints",allow_pickle=False)

    with torch.no_grad():
        image = vqvae.encode(image).latents 

    unet.to(device)
    vqvae.to(device)
    

    # generate gaussian noise to be decoded
    generator = torch.manual_seed(seed)
    noise = torch.randn(
        (image.size(0), unet.in_channels, unet.sample_size, unet.sample_size),
        generator=generator,
    ).to(device)


    # set inference steps for DDIM
    scheduler.set_timesteps(num_inference_steps=num_inference_steps)
    timesteps = torch.ones(image.size(0), device=device, dtype=torch.long)*scheduler.timesteps[int(len(scheduler.timesteps)*noise_percentage)]
    image = scheduler.add_noise(image,noise ,timesteps)
    scheduler.timesteps =scheduler.timesteps[int(len(scheduler.timesteps)*noise_percentage)+1:]


    for t in tqdm.tqdm(scheduler.timesteps):
        # predict noise residual of previous image
        with torch.no_grad():
            residual = unet(image, t)["sample"]

        # compute previous image x_t according to DDIM formula
        prev_image = scheduler.step(residual, t, image, eta=0.0)["prev_sample"]

        # x_t-1 -> x_t
        image = prev_image

    # decode image with vae
    with torch.no_grad():
        image = vqvae.decode(image)['sample']

    # process image
    image_processed = image.cpu()
    image_processed = (image_processed*0.5 + 0.5).clamp(0, 1)
    return image_processed
    



if __name__ == "__main__":

    image = []
    for i in range(4):
        image.append(original_data_val[i]['X'].unsqueeze(dim=0).to(device))
    image = torch.vstack(image)
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).absolute().parent.parent))
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Union, List
from diffusers.pipelines.pipeline_utils import ImagePipelineOutput
from diffusers.utils.torch_utils import randn_tensor
from diffusers import DDIMPipeline,DDPMPipeline
from diffusers.pipelines.pipeline_utils import DiffusionPipeline

import pytorch_lightning as pl
from pytorch_lightning import Callback
import matplotlib.pyplot as plt
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics import R2Score
import math
from torchmetrics import Metric
import itertools
from scipy.special import rel_entr
import numpy as np
import logging
from diffusers import AutoencoderKL


def process_query(unet, image, t,query, guidance_scale, null_token):
    """
    image: B x C x H x W
    query: B x Q x D
    guidance_scale: Union[float,int,torch.Tensor]
    null_token: B x D
    """
    #guidance scale can be a scalar or a list of scalars
    if isinstance(guidance_scale, Union[float,int]):
        model_in = torch.cat([image, image], 0)
        attr_in = torch.cat([null_token, query], 0)
        model_output = unet(model_in, t, attr_in)
        model_output_uncond, model_output_cond = model_output.chunk(2, dim=0)
        model_output = (1 - guidance_scale) * model_output_uncond + guidance_scale * model_output_cond
        return model_output
    elif isinstance(guidance_scale, list):
        if len(guidance_scale) != query.size(1):
            raise ValueError("guidance scale should be same as class labels")
        B,Q,D = query.size()
        model_in = torch.cat([image]*(Q+1), 0)
        query = query.transpose(1, 0).reshape(B*Q, D)
        query = torch.cat([null_token, query], 0)
        model_output = unet(model_in, t, query)
        chunk_model_output = model_output.chunk(Q+1, dim=0)
        model_output = chunk_model_output[0] + sum([guidance_scale[i]*(chunk_model_output[i+1] - chunk_model_output[0]) for i in range(Q)])
        return model_output
    else:
        raise ValueError("guidance scale should be a scalar or a tensor")


class CondDDIMPipeline(DDIMPipeline):
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
        vae: Optional[AutoencoderKL] = None,
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
        if query is not None:
            image = image.to(device=query.device)

        for t in self.progress_bar(self.scheduler.timesteps):
            
            # 2. predict previous mean of image x_t-1 and add variance depending on eta
            # eta corresponds to η in paper and should be between [0, 1]
            # do x_t -> x_t-1
            #train_timesteps 
            model_output = process_query(self.unet, image, t,query, guidance_scale, null_token)

            image = self.scheduler.step(
                model_output, t, image, eta=eta, use_clipped_model_output=use_clipped_model_output, generator=generator
            ).prev_sample

        if vae is not None:
            image = vae.decode(image/ vae.config.scaling_factor)[0]
        if not return_dict:
            return (image,)
        return ImagePipelineOutput(images=image)
    






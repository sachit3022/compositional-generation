
import torch
from torch import nn as nn
import numpy as np
import pytorch_lightning as pl
from diffusers import DDPMScheduler
from torch.nn import functional as F
from typing import  Union, List, Optional
from diffusers import UNet2DModel, VQModel, AutoencoderKL

class ComposableDiffusion(pl.LightningModule):

    def __init__(self, 
                model: UNet2DModel,
                noise_scheduler: DDPMScheduler,           
                vae: Optional[Union[VQModel,AutoencoderKL]] = None,
                lambda_coind: float = 0.0,
                  **kwargs):
        super().__init__()
        self.save_hyperparameters(ignore=['model','noise_scheduler','sampling_pipe','vae'])
        self.model = model
        self.noise_scheduler = noise_scheduler
        self.lambda_coind = lambda_coind
        self.vae = vae
        if self.vae is not None:
            for param in self.vae.parameters():
                param.requires_grad = False
    
    def prepare_labels(self,y,y_null):
        """
        y: (batch_size, num_cols)
        y_null: (batch_size, num_cols)
        returns: [(batch_size, num_cols), (batch_size*4, num_cols)]
            y_diffusion_obj: [x% of batch with y and 1-x% of y_null] or [x% of (batch_size, num_cols) with y[i] and 1-x% of y_null[i]]
            y_coind_obj: [y,y_null,y_i,y_-i]
        """
        p_null = 0.3
        masking = 'batch' # 'batch' or 'sample'
        coind_masking = 'pairwise' #, 'random'( 0.1 ), 'one vs all'


        if masking == 'batch':
            y_diffusion_obj = torch.where(torch.rand(y.size(0),1,device=y.device) < p_null , y_none, y)
        else:
            y_diffusion_obj = torch.where(torch.rand(*y.shape,device=y.device) < p_null , y_none, y)
            
            
        
        batch_size, num_cols = y.size()
        all_y_idx = torch.arange(num_cols).repeat(batch_size,1)
        y_indices = torch.argsort(torch.rand(*all_y_idx.shape[:2]), dim=1)
        y_idx = torch.gather(all_y_idx, dim=-1, index=y_indices)[:,:2]

        y_coind_obj = y.clone().repeat(4,1)
        x_idx = torch.arange(batch_size*4)
        y_ind[x_idx[:batch_size],y_idx[:batch_size,0]] = y_none[x_idx[:batch_size],y_idx[:batch_size,0]]
        y_ind[x_idx[batch_size:batch_size*3],:] = y_none[x_idx[:batch_size*2],:]
        y_ind[x_idx[batch_size:batch_size*2],y_idx[:batch_size,0]] = y[x_idx[:batch_size],y_idx[:batch_size,0]]    

        return y_diffusion_obj,y_coind_obj

    
    def training_step(self, batch, batch_idx):
        
        x0,y,y_null = batch['X'],batch['label'],batch['null_label']

        if self.vae is not None:
            x0 = self.vae.encode(x0)[0].mode()
            x0 = x0 * self.vae.config.scaling_factor
            
        
        y_diffusion_obj,y_coind_obj = self.prepare_labels(y,y_null)

        noise = torch.randn_like(x0)
        batch_size = x0.size(0)
        timesteps = torch.randint(
            0, self.noise_scheduler.num_train_timesteps,
            (batch_size,),
            device=self.device
        ).long()

        xt = self.noise_scheduler.add_noise(x0, noise, timesteps)
        noise_pred = self.model(xt, timesteps,y_diffusion_obj)
        l_diffusion  = F.mse_loss(noise_pred, noise)
        l_coind = 0.0

        ################### INDEPENDENCE ############################

        batch_size, num_cols = y.size()
        xt = xt.repeat(4,1,1,1) #replaced 4 with 2
        timesteps = timesteps.repeat(4)
        if self.lambda_coind > 0.0:    
            noise_pred_new = self.model(xt, timesteps,y_ind).chunk(4,dim=0)
            l_coind = F.mse_loss(noise_pred_new[0]+noise_pred_new[1], noise_pred_new[2]+noise_pred_new[3])
            l = l_diffusion + self.lambda_coind*l_coind
        else:
            with torch.no_grad():
                noise_pred_new = self.model(xt, timesteps,y_ind).chunk(4,dim=0)
                l_coind = F.mse_loss(noise_pred_new[0]+noise_pred_new[1], noise_pred_new[2]+noise_pred_new[3])
            l = l_diffusion
        
        ################### END INDEPENDENCE ############################
        self.log_dict({'diffusion_loss': l_diffusion, 'train_loss': l,  'coind_loss':l_coind}, prog_bar=True,on_epoch=True,sync_dist=True)
        return {'loss': l}
    
    def configure_optimizers(self):
        optimizer = self.hparams.optimizer(self.model.parameters())
        scheduler = self.hparams.scheduler(optimizer)
        return [optimizer],[{"scheduler": scheduler, "interval": "step"}]
    
    def validation_step(self,batch,batch_idx):
        x0,y,y_null = batch['X'],batch['label'],batch['null_label']

        if self.vae is not None:
            x0 = self.vae.encode(x0)[0].mode()
            x0 = x0 * self.vae.config.scaling_factor

        y_diffusion_obj,y_coind_obj = self.prepare_labels(y,y_null)

        noise = torch.randn_like(x0)
        batch_size = x0.size(0)
        timesteps = torch.randint(
            0, self.noise_scheduler.num_train_timesteps,
            (batch_size,),
            device=self.device
        ).long()
        xt = self.noise_scheduler.add_noise(x0, noise, timesteps)
        noise_pred = self.model(xt, timesteps,y_diffusion_obj)

        with torch.no_grad():
            l_diffusion  = F.mse_loss(noise_pred, noise)
            noise_pred_new = self.model(xt, timesteps,y_ind).chunk(4,dim=0)
            l_coind = F.mse_loss(noise_pred_new[0]+noise_pred_new[1], noise_pred_new[2]+noise_pred_new[3])
            l = l_diffusion + self.lambda_coind*l_coind
        
        self.log_dict({'diffusion_loss': l_diffusion, 'val_loss': l,  'coind_loss':l_coind}, prog_bar=True,on_epoch=True,sync_dist=True)
        return {'loss': l}


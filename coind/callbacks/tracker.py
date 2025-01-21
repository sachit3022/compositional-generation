import sys
from pathlib import Path
sys.path.append(str(Path(__file__).absolute().parent.parent))
from pytorch_lightning.callbacks import Callback
import torch
import pytorch_lightning as pl
import os
import matplotlib.pyplot as plt
from utils import save_images
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from typing import Optional
from diffusers import AutoencoderKL
from metrics import JSD, CS
from cs_classifier.models import MultiLabelClassifier
import torchvision

def training_grid_save(true_images, generated_images, save_path):
    #log_images to a file
    fig, (ax1, ax2) = plt.subplots(1, 2)
    save_images(true_images.detach().cpu(), path=ax1, title='true_samples')
    save_images(generated_images.detach().cpu(), path=ax2, title='conditional_generated_samples')
    fig.savefig(save_path)
    plt.close(fig)


class TrainingTracker(Callback):
    def __init__(self, 
                    sampling_pipe: DiffusionPipeline,
                    num_inference_steps: int = 50,
                    log_interval: int = 1,
                    num_classes_per_label: int =[10,10],
                    guidance_scale: float = 7.5,
                    output_dir: str = 'samples',
                    vae: Optional[AutoencoderKL] = None):
            super().__init__()
            self.sampling_pipe_partial = sampling_pipe
            self.num_inference_steps = num_inference_steps
            self.output_dir = output_dir
            if not os.path.exists(self.output_dir):
                os.makedirs(self.output_dir)
            self.log_interval = log_interval
            self.vae = vae
            self.guidance_scale = guidance_scale
            self.classifer_checkpoint = '/research/hal-gaudisac/Diffusion/compositional-generation/outputs/2025-01-20/15-12-41/classifier/version_0/checkpoints/epoch=21-step=10000.ckpt'

    def setup(self, trainer, pl_module, stage):
        if stage == 'fit':            
            self.sampling_pipe = self.sampling_pipe_partial(unet = pl_module.model,scheduler=pl_module.noise_scheduler)
            if self.vae is not None:
                self.vae = self.vae.to(pl_module.device)
                output_shape = (self.vae.config.in_channels,self.vae.config.sample_size,self.vae.config.sample_size)
            else:
                output_shape = (pl_module.model.config.in_channels,pl_module.model.config.sample_size,pl_module.model.config.sample_size)
            self.jsd_metric = JSD([10,10]).to(pl_module.device)

            model = MultiLabelClassifier(base_model= torchvision.models.resnet18(),num_classes_per_label=[10,10])
            model.load_state_dict(torch.load(self.classifer_checkpoint)["state_dict"])
            self.val_cs_metric_logger = CS(classifier = model).to(pl_module.device)
        

    def on_train_batch_start(self,
                                  trainer: pl.Trainer,
                                  pl_module: pl.LightningModule,
                                  batch: torch.Tensor,
                                  batch_idx: int,
                                  dataloader_idx: int =0):
        if batch_idx ==0 and (trainer.current_epoch +1)% self.log_interval == 0:
            save_path = f'{self.output_dir}/cond_gen_train_{trainer.current_epoch}.png'
            self.and_metric_tracking(batch,save_path=save_path,metric_logger=None)
           
    

    def on_validation_batch_start(self,
                                  trainer: pl.Trainer,
                                  pl_module: pl.LightningModule,
                                  batch: torch.Tensor,
                                  batch_idx: int,
                                  dataloader_idx: int =0):
        if batch_idx ==0  and (trainer.current_epoch +1)% self.log_interval == 0:
            save_path = f'{self.output_dir}/cond_gen_val_{trainer.current_epoch}.png'
            self.and_metric_tracking(batch,save_path=save_path,metric_logger=self.val_cs_metric_logger)
            self.jsd_metric.update(batch, pl_module.model,pl_module.noise_scheduler) #Confirmity score and JSD
    
    def and_metric_tracking(self, batch, save_path,metric_logger):
        """
        We focus on AND composition for tracking the training metrics
        """
        x,y,null_token  = batch["X"], batch["label"], batch["label_null"]    
        B,Q = y.size()
        query = null_token.repeat(Q,1).to(dtype=y.dtype,device=y.device)
        query = query.reshape(B,Q,Q)
        for i in range(Q):
            query[:,i,i] = y[:,i]
        guidance_scale = [self.guidance_scale]*Q
    
        generated_images = self.sampling_pipe(batch_size= x.size(0), 
                                num_inference_steps= self.num_inference_steps,
                                return_dict= True,
                                use_clipped_model_output = True,
                                query = query,
                                vae = self.vae,
                                guidance_scale=guidance_scale,
                                null_token=null_token)[0]
        
        if self.vae is not None:
            x = self.vae.decode(x/ self.vae.config.scaling_factor)[0]
        training_grid_save(x, generated_images, save_path)
        if metric_logger is not None:
            metric_logger.update(generated_images,query,null_token,guidance_scale)
        return 
    
    def on_validation_epoch_end(self, trainer, pl_module):
        if (trainer.current_epoch +1)% self.log_interval == 0:
            for metric_name, metric in self.jsd_metric.compute().items():
                pl_module.log(f'val/{metric_name}', metric, prog_bar=True,on_epoch=True,sync_dist=True)
            for metric_name, metric in self.val_cs_metric_logger.compute().items():
                pl_module.log(f'val/{metric_name}', metric, prog_bar=True,on_epoch=True,sync_dist=True)
            self.jsd_metric.reset()
            self.val_cs_metric_logger.reset()
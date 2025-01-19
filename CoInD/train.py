import os
import torch.random
import hydra
from hydra.utils import instantiate
from utils import set_seed
import pytorch_lightning as pl
import torch

@hydra.main(config_path='../configs',version_base='1.3',config_name='experiments/cmnist1d')
def main(cfg):
    
    ########## Hyperparameters and settings ##########
    set_seed(cfg.experiments.seed)
    ########## Dataset ##########

    logger = instantiate(cfg.experiments.logger)
    train_dataset = instantiate(cfg.experiments.data.train_dataset)
    val_dataset = instantiate(cfg.experiments.data.val_dataset)
   
    train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset,**cfg.experiments.data.train_dataloader)
    val_dataloader = torch.utils.data.DataLoader(dataset=val_dataset,**cfg.experiments.data.val_dataloader)

    ########## Callbacks and Logger ##########
    callbacks = []
    for callback_name, callback in cfg.experiments.callbacks.items():
        callbacks.append(instantiate(callback))
    logger = instantiate(cfg.experiments.logger)

    ########## Train the diffusion model #############
    diff_process  = instantiate(cfg.experiments.diffusion)
    diff_trainer = pl.Trainer(callbacks=callbacks,logger=logger,**cfg.experiments.trainer) 
    diff_trainer.fit(diff_process, train_dataloader, val_dataloader)
    
if __name__ == "__main__":  
    main()


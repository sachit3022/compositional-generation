import os
import torch.random
import hydra
from hydra.utils import instantiate
from utils import set_seed
import pytorch_lightning as pl
import torch


@hydra.main(config_path='../configs',version_base='1.3',config_name='experiments/cmnist_partial')
def main(cfg):
    
    ########## Hyperparameters and settings ##########
    set_seed(cfg.experiments.seed)
    ########## Dataset ##########

    logger = instantiate(cfg.experiments.logger)
    train_dataset = instantiate(cfg.experiments.dataset.train_dataset)
    val_dataset = instantiate(cfg.experiments.dataset.val_dataset)
   
    train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset,**cfg.experiments.dataset.train_dataloader)
    val_dataloader = torch.utils.data.DataLoader(dataset=val_dataset,**cfg.experiments.dataset.val_dataloader)

    ########## Callbacks and Logger ##########
    callbacks,loggers = [],[]
    for callback_name, callback in cfg.experiments.callbacks.items():
        callbacks.append(instantiate(callback))
    for logger_name, logger in cfg.experiments.loggers.items():
        loggers.append(instantiate(logger))


    ########## Train the diffusion model #############
    diff_process  = instantiate(cfg.experiments.diffusion)
    diff_trainer = pl.Trainer(callbacks=callbacks,logger=loggers,**cfg.experiments.trainer) 
    diff_trainer.fit(diff_process, train_dataloader, val_dataloader)
    
if __name__ == "__main__":  
    main()


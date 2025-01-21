import pytorch_lightning as pl
import torch
from torch import nn
from torch.nn import functional as F
from metrics import MultiLabelAcc

class ClassifierTrainer(pl.LightningModule):
    def __init__(self,
                 model: nn.Module,
                 num_classes_per_label:list[int],
                 optimizer: torch.optim.Optimizer,
                 scheduler: torch.optim.lr_scheduler._LRScheduler):
        super().__init__()
        
        self.save_hyperparameters(ignore=['model'])
        self.model = model
        self.val_acc = MultiLabelAcc(num_classes_per_label)
        self.train_acc = MultiLabelAcc(num_classes_per_label)
        self.num_classes_per_label = num_classes_per_label
    
    def configure_optimizers(self):
        """
        optimizer and scheduler
        """
        optimizer = self.hparams.optimizer(self.model.parameters())
        scheduler = self.hparams.scheduler(optimizer)
        return [optimizer],[{"scheduler": scheduler, "interval": "step"}]

    def training_step(self,batch,batch_idx):
        x,y = batch['X'],batch['label']
        logits = self.model(x)
        loss = 0
        for i in range(len(self.num_classes_per_label)):
            loss += F.cross_entropy(logits[i],y[:,i])
        self.train_acc.update(logits,y)

        self.log('train_loss',loss)
        return loss
    
    def validation_step(self,batch,batch_idx):
        x,y = batch['X'],batch['label']
        logits = self.model(x)
        loss = 0
        for i in range(len(self.num_classes_per_label)):
            loss += F.cross_entropy(logits[i],y[:,i])
        self.val_acc.update(logits,y)   
        return loss
    def on_validation_epoch_end(self):
        for i in range(len(self.num_classes_per_label)):
            self.log(f'val_accuracy_{i}',self.val_acc[i].compute(), prog_bar=True,on_epoch=True,sync_dist=True)
        self.val_acc.reset()
        return {}
    def on_train_epoch_end(self):
        for i in range(len(self.num_classes_per_label)):
            self.log(f'train_accuracy_{i}',self.train_acc[i].compute(),on_epoch=True,sync_dist=True)
        self.train_acc.reset()
        return {}
    def on_save_checkpoint(self, checkpoint):
        #save only the model
        checkpoint['state_dict'] = self.model.state_dict()
        return checkpoint
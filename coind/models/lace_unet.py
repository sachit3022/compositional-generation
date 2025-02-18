from diffusers.models.embeddings import TimestepEmbedding
from diffusers import UNet2DModel
from torch import nn
from typing import Tuple, Literal
import torch
from collections import OrderedDict

try:
    import xformers
    _XFORMERS_AVAILABLE = True
except Exception:
    _XFORMERS_AVAILABLE = False


class AttributeSpecificModel(UNet2DModel):
    def __init__(self,num_classes,**kwargs):
        super().__init__(**kwargs)
        self.class_embedding = None
        time_embed_dim = self.config.block_out_channels[0] * 4
        self.class_embedding = nn.Embedding(num_classes+1,time_embed_dim)
        if _XFORMERS_AVAILABLE:
            self.enable_xformers_memory_efficient_attention()
            
    def forward(self, x,t,y=None):
        """
        to remove .sample
        """
        return super().forward(x,t,y).sample



class ComposableUnet(nn.Module):
    def __init__(self,num_class_per_label,**kwargs):
        super().__init__()
        self.models = nn.ModuleList([AttributeSpecificModel(num_classes = num_classes,**kwargs) for num_classes in num_class_per_label])
        self.num_classes_per_label = num_class_per_label
        
    def forward(self,x,t,y=None):
        """
        x: B x C x H x W
        return : B x num_class_per_label x C x H x W
        """
        #split along the batch dimension
        y = torch.split(y,1,dim=1)
        #concatinate along batch dimension
        y = torch.cat([model(x,t,y_.view(-1)).unsqueeze(dim=1) for model,y_ in zip(self.models,y)],dim=1)
        return y
    @property
    def device(self):
        return self.models[0].device
    @property
    def dtype(self):
        return self.models[0].dtype
    @property
    def config(self):
        return self.models[0].config
            
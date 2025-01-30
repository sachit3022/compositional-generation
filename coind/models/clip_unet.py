import torch
from torch import nn
from torch.nn import functional as F
from typing import Tuple, Literal
from collections import OrderedDict

from diffusers.models import UNet2DModel
from dotmap import DotMap

try:
    import xformers
    _XFORMERS_AVAILABLE = True
except Exception:
    _XFORMERS_AVAILABLE = False




class MultiLabelEncoder(nn.Module):
    def __init__(self,num_class_per_label:Tuple[int, int],d_latent,interaction:Literal['cat','sum']) -> None:
        super().__init__()
        if interaction not in ['cat','sum']:
            raise ValueError(f"interaction must be either 'cat' or 'sum' not {interaction}")
        if interaction == 'cat':
            if d_latent % len(num_class_per_label) != 0:
                raise ValueError(f"d_latent {d_latent} must be divisible by len(num_class_per_label) {len(num_class_per_label)}")
            d_latent = d_latent//len(num_class_per_label)
        self.emb = nn.ModuleList([nn.Embedding(num_classes+1,d_latent) for num_classes in num_class_per_label])
        self.interaction = interaction

    def forward(self,y):
        if self.interaction == 'cat':
            y = torch.cat([emb(y[:,i]) for i,emb in enumerate(self.emb)],dim=1)
        elif self.interaction == 'sum':
            y = sum([emb(y[:,i]) for i,emb in enumerate(self.emb)])

        return y
        
class ClipUnet(UNet2DModel):
    def __init__(self,num_class_per_label,interaction='cat',**kwargs):
        super().__init__(**kwargs)
        self.class_embedding = None
        time_embed_dim = self.config.block_out_channels[0] * 4
        self.class_embedding = nn.Sequential(OrderedDict([('label_embedding', MultiLabelEncoder( num_class_per_label = num_class_per_label,
                                                                                                interaction = interaction,
                                                                 d_latent = time_embed_dim))
                                             ]))
        if _XFORMERS_AVAILABLE:
            self.enable_xformers_memory_efficient_attention()
            
    def forward(self, x,t,y=None):
        """
        to remove .sample
        """
        return super().forward(x,t,y).sample

class ProjUnetConditional(nn.Module):
    def __init__(self,input_dim= 768, num_class_per_label= [2,2],interaction='cat',**kwargs):
        """
        Suggestion from Mashrur.
        """
        super().__init__()
        self.dim_t = 1024
        self.proj = nn.Linear(input_dim, self.dim_t)
        self.config = DotMap({
                                'in_channels': 0,
                                'sample_size': (input_dim,)
                            })
        self.unet =  ClipUnet(num_class_per_label=num_class_per_label,interaction=interaction,**kwargs)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.proj_back = nn.Linear(self.dim_t,input_dim)
        self.dtype = torch.float32
    def forward(self,x,timesteps,cond):
        if timesteps.dim() == 0:
            timesteps = torch.full((x.size(0),), timesteps, dtype=torch.long, device=x.device)
        return self.proj_back(self.unet(self.proj(x).reshape(-1,1,32,32),timesteps,cond).reshape(x.size(0),-1))

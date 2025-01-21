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


class ClassConditionalUnet(UNet2DModel):
    def __init__(self,num_class_per_label,interaction='cat',**kwargs):
        super().__init__(**kwargs)
        self.class_embedding = None
        time_embed_dim = self.config.block_out_channels[0] * 4
        if interaction in ['cat','sum']:
            self.class_embedding = MultiLabelEncoder(num_class_per_label = num_class_per_label,
                                                        interaction = interaction,
                                                        d_latent = time_embed_dim)
        else:
            self.class_embedding = TimestepEmbedding(
                in_channels=len(num_class_per_label),
                time_embed_dim=time_embed_dim
            )
        if _XFORMERS_AVAILABLE:
            self.enable_xformers_memory_efficient_attention()
            
    def forward(self, x,t,y=None):
        """
        to remove .sample
        """
        return super().forward(x,t,y).sample
    

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
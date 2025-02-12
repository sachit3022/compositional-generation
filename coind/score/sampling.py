from typing import List, Tuple, Union
import torch
from torch import nn
from torch.nn import functional as F
from diffusers import UNet2DModel


class ModelWrapper(UNet2DModel):
    def __init__(self, model):
        super().__init__()
        self.model = model
    @torch.no_grad()
    def forward(self, x, t, y):
        raise NotImplementedError
    #config should be same as the model
    @property
    def config(self):
        return self.model.config


class ANDquery(ModelWrapper):

    def forward(self, x, t, y, **kwargs):
        # query, guidance_scale, null_token
        
        if 'null_token' not in kwargs:
            raise ValueError('null_token is not provided')    
        
        null_token = kwargs['null_token']
        query = self.prepare_and_query(y,null_token)
        B,Q,D = query.size()

        if 'guidance_scale' in kwargs:
            if isinstance(kwargs['guidance_scale'], float):
                guidance_scale = [kwargs['guidance_scale']]*Q
            else:
                guidance_scale = kwargs['guidance_scale']
        else:
            guidance_scale = [7.5]*Q

        model_in = torch.cat([x]*(Q+1), 0)
        query = query.transpose(1, 0).reshape(B*Q, D)
        query = torch.cat([null_token, query], 0)
        model_output = self.model(model_in, t, query)
        chunk_model_output = model_output.chunk(Q+1, dim=0)
        model_output = chunk_model_output[0] + sum([guidance_scale[i]*(chunk_model_output[i+1] - chunk_model_output[0]) for i in range(Q)])
        return model_output
    def prepare_and_query(self,y,null_token):
        B,Q = y.size()
        query = null_token.repeat(Q,1).to(dtype=y.dtype,device=y.device)
        query = query.reshape(B,Q,Q)
        for i in range(Q):
            query[:,i,i] = y[:,i]
        return query

class CFGquery(ModelWrapper):

    def forward(self, x, t, y, **kwargs):
        # query, guidance_scale, null_token
        
        if 'null_token' not in kwargs:
            raise ValueError('null_token is not provided')    
        
        null_token = kwargs['null_token']

        guidance_scale = kwargs.get('guidance_scale', 7.5)

        model_in = torch.cat([x]*(2), 0)
        query = torch.cat([null_token, y], 0)
        model_output = self.model(model_in, t, query)
        chunk_model_output = model_output.chunk(2, dim=0)
        model_output = chunk_model_output[0] + guidance_scale*(chunk_model_output[1] - chunk_model_output[0])
        return model_output
    def prepare_and_query(self,y,null_token):
        B,Q = y.size()
        query = null_token.repeat(Q,1).to(dtype=y.dtype,device=y.device)
        query = query.reshape(B,Q,Q)
        for i in range(Q):
            query[:,i,i] = y[:,i]
        return query
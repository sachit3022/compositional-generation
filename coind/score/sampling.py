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
    def forward(self, x, t, y, **kwargs):
        # query, guidance_scale, null_token
        
        if 'null_token' not in kwargs:
            raise ValueError('null_token is not provided')    
        
        null_token = kwargs['null_token']
        query,guidance_scale = self.prepare_query(y,null_token, kwargs.get('guidance_scale', 7.5))
        B,Q,D = query.size()
        model_in = torch.cat([x]*(Q+1), 0)
        query = query.transpose(1, 0).reshape(B*Q, D)
        query = torch.cat([null_token, query], 0)
        model_output = self.model(model_in, t, query)
        chunk_model_output = model_output.chunk(Q+1, dim=0)
        model_output = chunk_model_output[0] + sum([guidance_scale[i]*(chunk_model_output[i+1] - chunk_model_output[0]) for i in range(Q)])
        return model_output
    #config should be same as the model
    @property
    def config(self):
        return self.model.config
    def prepare_query(self,y,null_token,guidance_scale):
        raise NotImplementedError


class ANDquery(ModelWrapper):
    def prepare_query(self,y,null_token,guidance_scale):
        B,D = y.size()
        query = null_token.repeat(D,1).to(dtype=y.dtype,device=y.device)
        query = query.reshape(B,D,D)
        for i in range(D):
            query[:,i,i] = y[:,i]
        if isinstance(guidance_scale, float):
            guidance_scale = [guidance_scale]*D
        else:
            guidance_scale = guidance_scale
        return query,guidance_scale

class CFGquery(ModelWrapper):
    def prepare_query(self,y,null_token,guidance_scale):
        query = y.unsqueeze(dim=1)
        guidance_scale = [guidance_scale]*1
        return query,guidance_scale




class LaceModelWrapper(UNet2DModel):
    def __init__(self, model):
        super().__init__()
        self.model = model

    @torch.no_grad()
    def forward(self, x, t, y, **kwargs):
        # query, guidance_scale, null_token
        
        if 'null_token' not in kwargs:
            raise ValueError('null_token is not provided')    
        
        null_token = kwargs['null_token']
        query,guidance_scale = self.prepare_query(y,null_token, kwargs.get('guidance_scale', 7.5))
        B,Q,D = query.size()
        model_in = torch.cat([x]*(Q+1), 0)
        query = query.transpose(1, 0).reshape(B*Q, D)
        query = torch.cat([null_token, query], 0)
        model_output = self.model(model_in, t, query)
        chunk_model_output = model_output.chunk(Q+1, dim=0)
        model_output = chunk_model_output[0] + sum([guidance_scale[i]*(chunk_model_output[i+1] - chunk_model_output[0]) for i in range(Q)])
        return model_output.sum(dim=1)
    #config should be same as the model
    @property
    def config(self):
        return self.model.config
    def prepare_query(self,y,null_token,guidance_scale):
        raise NotImplementedError

class LaceANDquery(LaceModelWrapper):
    @torch.no_grad()
    def forward(self, x, t, y, **kwargs):
        # query, guidance_scale, null_token
        
        if 'null_token' not in kwargs:
            raise ValueError('null_token is not provided')    
        
        null_token = kwargs['null_token']
        query,guidance_scale = self.prepare_query(y,null_token, kwargs.get('guidance_scale', 7.5))
        B,Q,D = query.size()
        model_in = torch.cat([x]*(Q+1), 0)
        query = query.transpose(1, 0).reshape(B*Q, D)
        query = torch.cat([null_token, query], 0)
        model_output = self.model(model_in, t, query)
        chunk_model_output = model_output.chunk(2, dim=0)
        guidance_scale = guidance_scale.reshape(1,-1).repeat(B,1).reshape(B,-1,1,1,1)
        model_output = guidance_scale*(chunk_model_output[1] - chunk_model_output[0])
        return chunk_model_output[0].mean(dim=1) + model_output.sum(dim=1)

    def prepare_query(self,y,null_token,guidance_scale):
        query = y.unsqueeze(dim=1)
        guidance_scale = torch.tensor(guidance_scale,dtype=y.dtype,device=y.device)
        return query,guidance_scale

class LaceCFGquery(LaceModelWrapper):
    def prepare_query(self,y,null_token,guidance_scale):
        raise ValueError("CFG query is not valid for lace model because there is no joint distribution")



# class ModelWrapper(UNet2DModel):
#     def __init__(self, model):
#         super().__init__()
#         self.model = model
#     @torch.no_grad()
#     def forward(self, x, t, y):
#         raise NotImplementedError
#     #config should be same as the model
#     @property
#     def config(self):
#         return self.model.config


# class ANDquery(ModelWrapper):

#     def forward(self, x, t, y, **kwargs):
#         # query, guidance_scale, null_token
        
#         if 'null_token' not in kwargs:
#             raise ValueError('null_token is not provided')    
        
#         null_token = kwargs['null_token']
#         query = self.prepare_and_query(y,null_token)
#         B,Q,D = query.size()

#         if 'guidance_scale' in kwargs:
#             if isinstance(kwargs['guidance_scale'], float):
#                 guidance_scale = [kwargs['guidance_scale']]*Q
#             else:
#                 guidance_scale = kwargs['guidance_scale']
#         else:
#             guidance_scale = [7.5]*Q

#         model_in = torch.cat([x]*(Q+1), 0)
#         query = query.transpose(1, 0).reshape(B*Q, D)
#         query = torch.cat([null_token, query], 0)
#         model_output = self.model(model_in, t, query)
#         chunk_model_output = model_output.chunk(Q+1, dim=0)
#         model_output = chunk_model_output[0] + sum([guidance_scale[i]*(chunk_model_output[i+1] - chunk_model_output[0]) for i in range(Q)])
#         return model_output
#     def prepare_and_query(self,y,null_token):
#         B,Q = y.size()
#         query = null_token.repeat(Q,1).to(dtype=y.dtype,device=y.device)
#         query = query.reshape(B,Q,Q)
#         for i in range(Q):
#             query[:,i,i] = y[:,i]
#         return query

# class CFGquery(ModelWrapper):

#     def forward(self, x, t, y, **kwargs):
#         # query, guidance_scale, null_token
        
#         if 'null_token' not in kwargs:
#             raise ValueError('null_token is not provided')    
        
#         null_token = kwargs['null_token']

#         guidance_scale = kwargs.get('guidance_scale', 7.5)

#         model_in = torch.cat([x]*(2), 0)
#         query = torch.cat([null_token, y], 0)
#         model_output = self.model(model_in, t, query)
#         chunk_model_output = model_output.chunk(2, dim=0)
#         model_output = chunk_model_output[0] + guidance_scale*(chunk_model_output[1] - chunk_model_output[0])
#         return model_output
#     def prepare_and_query(self,y,null_token):
#         B,Q = y.size()
#         query = null_token.repeat(Q,1).to(dtype=y.dtype,device=y.device)
#         query = query.reshape(B,Q,Q)
#         for i in range(Q):
#             query[:,i,i] = y[:,i]
#         return query
import torch
from torch.nn import MultiheadAttention
from torch import tensor

class temporal_encoder(torch.nn.Module) :
    def __init__(self):
        super().__init__()

    def forward(self,node_history:dict):
        recent = tensor(node_history[-1])

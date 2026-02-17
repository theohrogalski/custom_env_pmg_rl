import torch
from torch.nn import Module

class uncertainty_estimator(Module):
    def __init__(self):
        super().__init__()
        self.data=[]
        self.network = torch.nn.Sequential(torch.nn.Linear(1,10),
                                           torch.nn.ReLU(),
                                           torch.nn.Linear(10,1)
                                           )
    def forward(self):
        return self.network(self.data)


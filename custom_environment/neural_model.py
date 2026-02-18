import torch
from torch.nn import Module
import networkx as nx
from torch_geometric.nn import GCNConv
class uncertainty_estimator(Module):
    def __init__(self, feature_dim,hidden_dim,out_dim):
        super().__init__()
        self.data=[]
        
        self.gcnconv1 = (GCNConv(feature_dim,hidden_dim))
        self.gcnconv2 = (GCNConv(feature_dim,hidden_dim))

        self.lin =  torch.nn.Linear(feature_dim, hidden_dim,out_dim)
                          
        
        self.optimizer = torch.optim.Adam(self.parameters(),lr=1e-3)
        self.loss_f = torch.nn.MSELoss()

    def forward(self,x,edge_index):
        assert x.shape == torch.Size([50,5])
        x = self.gcnconv1(x, edge_index)
        print(f"here is {x.shape}")
        x = torch.nn.functional.relu(x)
        print(f"here2 is {x.shape}")

        x = self.gcnconv2(x, edge_index)

        x = torch.nn.functional.relu(x)
        
        return self.lin(x)
    
    def update_estimator(self, x, edge_index):
    # Enable gradients explicitly in case the parent loop turned them off
        with torch.enable_grad():
            self.optimizer.zero_grad()
            
            # Ensure the model is in training mode
            self.train() 
            
            prediction = self.forward(x, edge_index)
            target = x.detach() 
            
            loss = self.loss_f(prediction, target)
            loss.backward()
            self.optimizer.step()
        return loss.item()
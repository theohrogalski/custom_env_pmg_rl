import torch
from torch.nn import Module
import networkx as nx
from torch_geometric.nn import GCNConv
class uncertainty_estimator(Module):
    def __init__(self, feature_dim,hidden_dim,out_dim):
        super().__init__()

        if torch.cuda.is_available():
            print("cuda2")
            self.device="cuda"
        else: 
            self.device="cpu"
        self.data=[]
        
        self.gcnconv1 = (GCNConv(feature_dim,out_channels=5))
        self.gcnconv2 = (GCNConv(feature_dim,out_channels=5))

        self.lin =  torch.nn.Linear(feature_dim, out_features=1,device=self.device)
                          
        
        self.optimizer = torch.optim.Adam(self.parameters(),lr=1e-3)
        self.loss_f = torch.nn.MSELoss()

    def forward(self,x,edge_index):
        #print(type(edge_index))
        edge_index = edge_index.to(self.device)
        #print(edge_index.device)

        x= x.to(self.device)
        #print(x.device)
        assert x.shape == torch.Size([50,5])
        #print(x.shape)
        x = self.gcnconv1(x, edge_index)
        #print(f"here is {x.shape}")
        #print(f"shape here is {x.shape}")
        x = torch.nn.functional.relu(x)

        #print(f"here2 is {x.shape}")

        x = self.gcnconv2(x, edge_index)

        x = torch.nn.functional.relu(x)
        return self.lin(x)
    
    def update_estimator(self, x, edge_index):
        x.to(self.device)
        edge_index.to(self.device)
    # Enable gradients explicitly in case the parent loop turned them off
        with torch.enable_grad():
            self.optimizer.zero_grad()
            
            # Ensure the model is in training mode
            self.train() 
            
            prediction = self.forward(x, edge_index)
            target = x.detach() 
            
            #print(prediction.shape)
            
            #print(target.shape)
            target = target[:,0].reshape(50,1)
            
            assert target.shape == torch.Size([50,1])
            loss = self.loss_f(prediction, target)
            loss.backward()
            self.optimizer.step()
        return loss.item()
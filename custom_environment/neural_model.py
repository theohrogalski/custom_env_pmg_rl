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
        self.num_nodes=20

            # optional second layer for better graph depth                  
        self.lin = torch.nn.Linear(2,1)
        

        self.optimizer = torch.optim.Adam(self.parameters(),lr=1e-3)
        self.loss_f = torch.nn.MSELoss()

    def forward(self,x,edge_index,move_num):
        #print(type(edge_index))

        edge_index = edge_index.to(self.device)
        x= x.to(self.device)
        x=x[:,0]
        move_num = torch.tensor(move_num).expand(self.num_nodes)
        move_num = move_num.to(self.device)
        #print(move_num.shape)
        #print(x.shape)
        x = torch.concat((x.unsqueeze(1),move_num.unsqueeze(1)),1)
        x=(self.lin(x))
        #print(f"shape of x here iss {x.shape}")
        return x
    
    def update_estimator(self, x, edge_index,move_num):
        
        # Ensure the model is in training mode
        self.train() 
        
        prediction = self.forward(x, edge_index,move_num)
        target = x.detach() 
        
        #print(prediction.shape)
        
        #print(target.shape)
        target = target[:,0].reshape(self.num_nodes,1)
        #print(f"target is {target}")
        #print(f" prediction is {prediction}")

        #assert target.shape == torch.Size([50,1])
        loss = self.loss_f(prediction, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()
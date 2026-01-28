import torch
from torch_geometric.nn.models import GAT
from torch_geometric.nn.conv import GATConv
from torch_geometric.utils import from_networkx
import networkx as nx
from torch_geometric.data import Data
from torch_geometric.nn.aggr import MLPAggregation
class observation_processing_network(torch.nn.Module):
    def __init__(self,number_of_nodes) :
        if torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"
        super().__init__()
        self.graph_multihead = GATConv(in_channels=-1,out_channels=10,heads=8)
        self.graph_attention = GAT(in_channels=-1,num_layers=10, hidden_channels=10)
        self.mlp_aggr = MLPAggregation(in_channels=10,out_channels=10,max_num_elements=80,num_layers=10, hidden_channels=10)
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(in_features=10,out_features=10),
            torch.nn.Sigmoid(),
            torch.nn.Linear(in_features=2,out_features=10),
            torch.nn.Sigmoid(),
            torch.nn.Linear(in_features=2, out_features=number_of_nodes)
            )
        self.softmax = torch.nn.Softmax()
        self.history = []
    def forward(self, mental_map:nx.Graph, mask:list):
        mental_map_data=from_networkx(mental_map)
        gat, _ = self.graph_attention(mental_map_data)
        (self.history).append(gat)
        print(self.history)
        mha, = self.graph_multihead(gat, torch(self.history))
        logits:torch.Tensor = self.mlp(results)
        for index, logit in enumerate(iter(logits)):
            if mask[index]==0:
                logits[index]=-1
        return logits
        


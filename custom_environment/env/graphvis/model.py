import torch
from torch_geometric.nn.models import GAT
from torch_geometric.nn.conv import GATConv
from torch_geometric.utils import from_networkx
import networkx as nx
from torch_geometric.data import Data
from temp_encoder import temporal_encoder
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

    def forward(self, mental_map:nx.Graph, mask:list):
        if mental_map:
            print("mental map exists")
            mental_map_tg_data = from_networkx(mental_map,group_node_attrs="all")
            if mental_map_tg_data.num_nodes>0:
                gat,_ = self.graph_attention(mental_map_tg_data)
        else:
            placeholder_x = torch.zeros((1)) 
            placeholder_edge_index = torch.empty((2, 0), dtype=torch.long)
            gat = Data(x=placeholder_x, edge_index=placeholder_edge_index)        
 
        history:list = []
        for maps in history:
            history.append(from_networkx(maps,group_node_attrs="all"))
        lists = tuple(history)

        if history:
            print("history exists")
            mha,_ = self.graph_multihead(gat,h,lists)
        else:
            print("history not existent")
            placeholder_x = torch.zeros((1)) 
            placeholder_edge_index = torch.empty((2, 0), dtype=torch.long)

            mha = Data(x=placeholder_x, edge_index=placeholder_edge_index)        

        logits:torch.Tensor = self.mlp(results) 
        print(f"logits are {logits}")
        for index, logit in enumerate(iter(logits)):
            if mask[index]==0:
                logits[index]=-1
        return logits
        


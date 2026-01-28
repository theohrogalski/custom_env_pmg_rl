import torch
from torch_geometric.nn.models import GAT
from torch_geometric.nn.conv import GATConv
from torch_geometric.utils import from_networkx
import networkx as nx
from torch.nn import MultiheadAttention
from torch_geometric.data import Data
from torch.nn import ReLU
from torch_geometric.nn.aggr import MLPAggregation
from torch_geometric.utils import add_self_loops
class observation_processing_network(torch.nn.Module):
    def __init__(self,number_of_nodes) :
        if torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"
        super().__init__()
        self.multihead = MultiheadAttention(embed_dim=8, num_heads=4)
        self.graph_attention = GAT(in_channels=-1,num_layers=10,hidden_channels=3)
        self.mlp_aggr = MLPAggregation(in_channels=10,out_channels=10,max_num_elements=80,num_layers=10, hidden_channels=10)
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(in_features=10,out_features=10),
            torch.nn.Sigmoid(),
            torch.nn.Linear(in_features=2,out_features=10),
            torch.nn.Sigmoid(),
            torch.nn.Linear(in_features=2, out_features=number_of_nodes)
            )
        self.softmax = torch.nn.Softmax()
        self.history=[]
    def forward(self, mental_map:nx.Graph, mask:list):
        if mental_map:
            mental_map = from_networkx(mental_map, group_node_attrs=["uncertainty","agent_presence","target"])
        else:
            mental_map = from_networkx(nx.cycle_graph(n=1))
        print(mental_map)
        mental_map.x = mental_map.x.to(torch.float32)
        mental_map.edge_index = add_self_loops(mental_map.edge_index)
        print(f"edge index is {mental_map.edge_index}")

        gat_x = self.graph_attention(mental_map.x,mental_map.edge_index[0])
        
        self.history.append(gat_x[0])
        
        
        results = self.multihead(gat_x[0], self.history,self.history)
        
        results = self.mlp_aggr(results)
        
        results = self.mlp(results)
        
        new_results = []
        
        for index, logit in enumerate(list(results)):
            if mask[index]==1:
                new_results.append(logit)
            else:
                new_results.append(0)
        print(f"logits are {new_results}")
        
        return new_results
        

# Test

if __name__ == "__main__":
    print("starting process")
    model = observation_processing_network(40)
    graph = nx.read_graphml("./graphs/int_name_graph.graphml")
    model.forward(graph,[1,1,1,1,1,1,1,1,1,1,1,1,0,1,1,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1])
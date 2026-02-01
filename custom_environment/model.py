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
        self.multihead = MultiheadAttention(embed_dim=3, num_heads=3)
        self.number_of_nodes=number_of_nodes
        custom_mlp = torch.nn.Sequential(torch.nn.Linear(3*number_of_nodes,out_features=16),
                                         torch.nn.ReLU(),
                                         torch.nn.Linear(16,32),
                                         torch.nn.ReLU(),
                                         torch.nn.Linear(in_features=16,out_features=40)
                                         )


        self.graph_attention = GAT(in_channels=-1,num_layers=10,hidden_channels=3)

        self.mlp_aggr = MLPAggregation(in_channels=3, out_channels=1,max_num_elements=3,num_layers=3, hidden_channels=5, mlp = custom_mlp)

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
        
        self.history.append(gat_x)
        
        history = torch.concatenate(self.history)

        
        #print(f"gat x 0 is  {gat_x}")
        mha_to_mlp_aggr_format = []
        results, _ = self.multihead(gat_x, history, history)
        print((results).shape)
        index=[]
        for i in range(self.number_of_nodes):
            index.append(i)
        index = torch.tensor(index,dtype=torch.int32)    
        results = self.mlp_aggr(x=results, index=index)

        new_results = [   ] 
        
        for index, logit in enumerate(list(results)):
            if mask[index]==1:
                new_results.append(logit[0])
            else:
                new_results.append(torch.tensor([0]))
        print(f"logits are {new_results}")
        print(f"new results are {new_results}")
        print()
        return new_results
        

# Test

if __name__ == "__main__":
    print("starting process")
    model = observation_processing_network(40)
    graph = nx.read_graphml("./graphs/int_name_graph.graphml")
    model.forward(graph,[0,1,1,1,1,1,1,1,1,1,1,1,0,1,1,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1])
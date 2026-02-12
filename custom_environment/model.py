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
                                         torch.nn.Linear(in_features=16,out_features=number_of_nodes)
                                         )
        self.transform = torch.nn.Transformer()
        self.graph_attention = GAT(in_channels=-1,num_layers=10,hidden_channels=3)

        self.actor = MLPAggregation(in_channels=3, out_channels=1,max_num_elements=3,num_layers=3, hidden_channels=5, mlp = custom_mlp)
        self.critic = torch.nn.Linear(in_features=1,out_features=1)
        self.softmax = torch.nn.Softmax()
        
        self.history=[]
    
    def forward(self, mental_map:nx.Graph, mask:list):
        #print(mental_map)
        if mental_map:
            """for _,data in mental_map.nodes(data=True):
                print(data)"""

            mental_map = from_networkx(mental_map, group_node_attrs=["uncertainty","agent_presence","target"])
        else:
            mental_map = (nx.cycle_graph(n=50))
            #print(mental_map)
            for node in mental_map.nodes():
                mental_map.nodes[node]["uncertainty"] = 0
                mental_map.nodes[node]["agent_presence"] = 0
                mental_map.nodes[node]["target"] = 0
            mental_map=from_networkx(mental_map, group_node_attrs=["uncertainty","agent_presence","target"])
        #print(mental_map)
        #print(mental_map.x)
        mental_map.x = mental_map.x.to(dtype=torch.float32)
        mental_map.edge_index = add_self_loops(mental_map.edge_index)

        gat_x = self.graph_attention(mental_map.x, mental_map.edge_index[0])
        
        # 1. Store only the DETACHED version in history to break the graph link
        self.history.append(gat_x.detach()) 
        
        # 2. Keep the history list from growing forever (optional but recommended)
        if len(self.history) > 10: 
            self.history.pop(0)

        # 3. Concatenate the detached history with the CURRENT active gat_x
        # Use gat_x for the current step and the history for context
        current_history = torch.concatenate(self.history[:-1] + [gat_x])

        # 4. Use current_history in your attention layer
        results, _ = self.multihead(gat_x, current_history, current_history)
        #print((results).shape)
        index=[]
        for i in range(self.number_of_nodes):
            index.append(i)
        index = torch.tensor(index,dtype=torch.int32) 
        #print(results)   
        results = self.actor(x=results, index=index)
        #print(results)
        #print(type(results))
        #print(mask)
        
        new_results = results[:,0] * torch.tensor(mask)       
        #print(f"new results are {new_results}")
        #print()
        value = self.critic(results)
        """print(type(new_results))
        print(new_results)
        print(new_results.sum())"""
        value = value.mean()
        #print(value)
        return new_results, value
        

#       Test

if __name__ == "__main__":
    print("starting process")
    model = observation_processing_network(40)
    graph = nx.read_graphml("./graphs/int_name_graph.graphml")
    model.forward(graph,[0,1,1,1,1,1,1,1,1,1,1,1,0,1,1,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1])

import torch
import torch.nn as nn
from torch_geometric.nn import GAT, TransformerConv, MLPAggregation
from torch_geometric.utils import from_networkx, get_laplacian, to_dense_adj, add_self_loops
import networkx as nx

class observation_processing_network(torch.nn.Module):
    def __init__(self, number_of_nodes):
        super().__init__()
        self.number_of_nodes = number_of_nodes
        self.device = "cpu"
        # 1. Feature Processing
        self.graph_attention = GAT(in_channels=5, hidden_channels=8, num_layers=2, out_channels=5)
        self.multihead = nn.MultiheadAttention(embed_dim=5, num_heads=1)
        self.transform_two = TransformerConv(5, 5, heads=1)

        # 2. Actor-Critic Heads
        # custom_mlp must match the flattened input of MLPAggregation
        custom_mlp = nn.Sequential(
            nn.Linear(5 * number_of_nodes, 64),
            nn.ReLU(),
            nn.Linear(64, number_of_nodes)
        )
        self.actor = MLPAggregation(in_channels=5, out_channels=number_of_nodes, mlp=custom_mlp,max_num_elements=number_of_nodes,num_layers=10,hidden_channels=10)
        self.critic = nn.Linear(number_of_nodes, 1)

    def compute_pyg_laplacian_features(self, the_data, k=2):
        # Ensure we stay on the same device
        edge_index, _ = get_laplacian(the_data.edge_index, normalization='sym', num_nodes=50)
        L = to_dense_adj(edge_index, max_num_nodes=50).squeeze(0)
        evals, evecs = torch.linalg.eigh(L)
        return evecs[:, :k], evals[1] # [50, 2], Fiedler Value

    def forward(self, mental_map_nx: nx.Graph, mask: list):
        # 1. Convert and Initial [50, 5] Construction
        data = from_networkx(mental_map_nx, group_node_attrs=["uncertainty", "agent_presence", "target"]).to(self.device)
        
        # Add Laplacian features [50, 2] to the [50, 3] raw features
        lap_ev, fiedler = self.compute_pyg_laplacian_features(data)
        x_combined = torch.cat([data.x, lap_ev], dim=1) # [50, 5]
        
        # 2. Graph Processing
        edge_index, _ = add_self_loops(data.edge_index, num_nodes=50)
        
        # GAT Layer
        x = self.graph_attention(x_combined, edge_index)
        
        # Attention (Self-attention on the nodes)
        # Multihead expects [Seq, Batch, Embed] -> [50, 1, 5]
        x_att = x.unsqueeze(1)
        attn_out, _ = self.multihead(x_att, x_att, x_att)
        x = attn_out.squeeze(1)
        
        # Transformer Layer
        x = self.transform_two(x, edge_index)

        # 3. Actor-Critic Output
        # index must be [50]
        idx = torch.arange(self.number_of_nodes, device=self.device)
        
        # MLPAggregation returns [1, number_of_nodes]
        logits = self.actor(x, index=idx)
        
        # Apply Mask (Safety/Valid actions)
        mask_tensor = torch.tensor(mask, device=self.device).float()
        masked_logits = logits.squeeze(0) * mask_tensor
        
        # Critic value
        value = self.critic(logits).mean()

        return masked_logits, value, x_combined, edge_index # Return x_combined for the Estimator!
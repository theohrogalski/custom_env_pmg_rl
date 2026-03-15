import torch
import torch.nn as nn
from torch_geometric.nn import GAT, TransformerConv, MLPAggregation
from torch_geometric.utils import from_networkx, get_laplacian, to_dense_adj, add_self_loops
import networkx as nx

class observation_processing_network(torch.nn.Module):
    def get_safe_logits(self, logits, x_state, edge_index, unc_net, threshold=0.8, eta=0.1):
        """
        logits: [50] tensor from the Actor
        x_state: [50, 5] features
        unc_net: The GCN (Architecture 1)
        """
        ###print(x_state.shape)
        ###print(logits.shape)
    
        with torch.no_grad():
            # 1. Get current 'Mental Map' from GCN
            # predicted_u shape: [50, 1]
            predicted_u = unc_net(x_state, edge_index)

            # 2. Calculate Current Safety h(x_t)
            h_t = threshold - torch.max(predicted_u)
            
            # 3. Look-ahead: Predict h(x_t+1) for each possible move
            # For simplicity in MARL, we check if moving to node 'j' 
            # reduces uncertainty enough to satisfy the DCBF.
            
            # We create a safety mask. 1.0 = Safe, 0.0 = Blocked
            safety_mask = torch.ones_like(logits)
            
            for node_idx in range(self.number_of_nodes):
                # Heuristic: If GCN thinks node 'i' is already near the limit,
                # and it's NOT the node we are moving to, the move might be unsafe.
                ###print(type(predicted_u[node_idx]))
                ###print(predicted_u[node_idx].shape)
                if predicted_u[node_idx].item() > (threshold - eta * h_t):
                    # If we don't move to this high-uncertainty node, 
                    # we risk violating the barrier.
                   logits[node_idx]+=0.1
            # 4. Apply the Barrier: If h_t is dropping too fast, 
            # we 'force' the logits toward the critical nodes.
            if h_t < 0.2: # Buffer zone
                critical_node = torch.argmax(predicted_u)
                # Projection: Zero out all other logits, or heavily bias the critical one
                #print(f"logits are {logits}")
                logits[critical_node] += 0.1
                
        return logits
    def __init__(self, number_of_nodes):
        
        super().__init__()
        self.number_of_nodes = number_of_nodes
        if torch.cuda.is_available():
            self.device="cuda"
            ###print("cuda")
        else:
            self.device="cpu"
        # 1. Feature Processing
        self.graph_attention = GAT(in_channels=6, hidden_channels=8, num_layers=2, out_channels=5)
        self.multihead = nn.MultiheadAttention(embed_dim=5, num_heads=1)
        self.transform_two = TransformerConv(5, 5, heads=1)

        # 2. Actor-Critic Heads
        # custom_mlp must match the flattened input of MLPAggregation
        custom_mlp = nn.Sequential(
            nn.Linear(number_of_nodes, 64),
            nn.ReLU(),
            nn.Linear(64, number_of_nodes)
        )
        self.actor = nn.Linear(self.number_of_nodes*5,self.number_of_nodes)
        self.critic = nn.Linear(in_features=self.number_of_nodes*3, out_features=1)

    def compute_pyg_laplacian_features(self, the_data, k=2):
        edge_index, _ = get_laplacian(the_data.edge_index, normalization='sym', num_nodes=self.number_of_nodes)
        L = to_dense_adj(edge_index, max_num_nodes=self.number_of_nodes).squeeze(0)
        evals, evecs = torch.linalg.eigh(L)
        return evecs[:, :k], evals[1] # [50, 2], Fiedler Value

    def forward(self, mental_map_nx: nx.Graph, mask: list,unc_net):
        
        data = from_networkx(mental_map_nx, group_node_attrs=["uncertainty", "agent_presence", "target"])
        # Add Laplacian features [50, 2] to the [50, 3] raw features
        lap_ev, fiedler = self.compute_pyg_laplacian_features(data)
        ###print(lap_ev.shape)
        data_x = (data.x).to(self.device).float()
        data_x=data_x.flatten()
        value = self.critic(data_x)
        #print(f"value is here {value}")
        
        x_combined = torch.cat([data.x, lap_ev], dim=1) # [50, 5]
        assert x_combined.shape == torch.Size([50,5])
        with torch.no_grad(): # Use no_grad here so Actor doesn't backprop through GCN

            uncertainty_prediction = unc_net(x_combined, data.edge_index) # [50, 1]
    
    # 5. ENRICH THE STATE: Add the prediction as a 6th feature
    # Now state is [50, 6] -> (Obs + Topology + Prediction)
        uncertainty_prediction = uncertainty_prediction.to(self.device)
        x_combined = x_combined.to(self.device)
        
        x_enriched = torch.cat([x_combined, uncertainty_prediction], dim=1)
        # 2. Graph Processing
        x_enriched = x_enriched.to(self.device)
        edge_index, _ = add_self_loops(data.edge_index, num_nodes=self.number_of_nodes)
        
        # GAT Layer

        x_enriched = x_enriched.to(self.device)
        edge_index = edge_index.to(self.device)

        x = self.graph_attention(x_enriched, edge_index)


        # Attention (Self-attention on the nodes)
        # Multihead expects [Seq, Batch, Embed] -> [50, 1, 5]
        x_att = x.unsqueeze(1)
        
        attn_out, _ = self.multihead(x_att, x_att, x_att)
        ##print(attn_out.shape)
        x = attn_out.squeeze(1)
        # Transformer Layer
        x = self.transform_two(x, edge_index)
        
        ##print(f"x is {x}")
        # 3. Actor-Critic Output
        # index must be [50]
        #idx = torch.arange(self.number_of_nodes, device=self.device)
        
        # MLPAggregation returns [1, number_of_nodes]
        
        logits = self.actor(x.flatten())
        #print(logits.shape)
        ##print(logits.shape)
        ##print(f"logit shape here is {logits.shape}")
        logits = self.get_safe_logits(logits,x,edge_index,unc_net)
        ##print(f"here logits shape are {logits.shape}")
        ##print(f"logit shape here is {logits.shape}")

        # Apply Mask (Safety/Valid actions)
        mask_tensor = torch.tensor(mask, device=self.device).float()
        ##print(mask_tensor.shape)
        ##print(f"logits shape before is {logits.shape}")
        ##print(f"logits squeeze shape {logits.squeeze(1).shape}")
        
        masked_logits = logits  *mask_tensor
        ##print(f"masked logits shape are {masked_logits.shape}")
        # Critic value
        
        return masked_logits, value, x_combined, edge_index 
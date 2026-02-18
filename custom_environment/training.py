from custom_graph_env import GraphEnv
import torch
import os
from torch.nn.modules.container import ParameterList
from torch.distributions import Categorical
from model_two import observation_processing_network
import logging
from torch.nn.functional import mse_loss
from gymnasium.wrappers import RecordEpisodeStatistics
logger = logging.getLogger("logger_train")
logging.basicConfig(filename='logger_train.log', level=logging.INFO)
print("logger created")
logger.info("------ Logger Started ------")
def calculate_unc_est_loss(predicted_value, actual_value):
    return (predicted_value-actual_value)^2

cur_length_list = []

def compute_ac_loss(log_prob, value, reward, next_value, done, gamma=0.99):
    # 1. Calculate Target (TD Target)
    # If done, next value is 0
    mask = 1 - int(done)
    target = reward + (gamma * next_value * mask)
    
    # 2. Calculate Advantage (Target - Baseline)
    # Detach target because we don't want to backprop through the target for the critic
    advantage = target - value
    
    # 3. Actor Loss: -log_prob * advantage
    actor_loss = -log_prob * advantage
    
    # 4. Critic Loss: MSE(value, target)
    # Use SmoothL1 or MSE
    critic_loss = mse_loss(torch.Tensor([value]), torch.Tensor([target]))
    
    # 5. Total Loss
    total_loss = actor_loss + (0.5 * critic_loss)
    
    return total_loss
num_nodes=50
env = GraphEnv(num_nodes=num_nodes)
#print(f" here3 {env.graph.nodes()}")
#print(env.agent_position)

obs_nets:dict = {agent:observation_processing_network(env.graph.number_of_nodes()) for agent in env.possible_agents}
optimizers = {agent:torch.optim.Adam(obs_nets[agent].parameters()) for agent in env.agents}
gamma = 0.99

critic_loss_dict:dict = {}
#print("starting")
import logging

# Hyperparameters
GAMMA = 0.99
critic_loss_dict = {}
# Main Episode Loop
while env.agents:
    
    actions={}
    step_data={}
    # --- PHASE 1: COLLECT ACTIONS ---
    for agent in env.agents:
    # 1. Run the forward pass
    # NOTE: We now get the [50, 5] state (x_state) back from the forward pass
        logits, value, x_state, edges = obs_nets[agent](env.mental_map[agent], env.action_mask_to_node[int(agent[6:])])

        # 2. UPDATE THE UNCERTAINTY ESTIMATOR (GCN)
        # We do this separately. Use .detach() so it doesn't break the Actor's graph.
        unc_net = env.agent_to_net[agent]
        # This call now only handles the GCN logic
        unc_loss = unc_net.update_estimator(x_state.detach(), edges)

        # 3. Handle RL sampling
        dist = Categorical(logits=logits)
        action_tensor = dist.sample()
        action_tensor=action_tensor.float()
        actions[agent] = action_tensor
        # Store for Phase 3
        step_data[agent] = {
            "log_prob": dist.log_prob(action_tensor),
            "value": value,
            "prediction": unc_net(x_state, edges).detach() # For Task 2 (DCBF)
        }
    # --- PHASE 2: STEP ENVIRONMENT ---
    # We step ONCE with all actions
    # Note: PettingZoo returns dicts keyed by agent
    
    obs, rewards, terminations, truncations, infos = env.step(actions)


        # Logging
        
    # Logging
    logging.info(f"Rewards: {list(rewards.values())}")
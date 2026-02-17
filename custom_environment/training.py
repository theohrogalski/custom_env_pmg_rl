from custom_graph_env import GraphEnv
import torch
import os
from torch.nn.modules.container import ParameterList
from torch.distributions import Categorical
from model import observation_processing_network
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
net_param_list = ParameterList()
for net in [env.agent_to_net.values()]:
    net_param_list.append(net.parameters())

neural_net_optim=torch.optim.Adam(params=net_param_list)

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
    
    # --- PHASE 1: COLLECT ACTIONS ---
    actions = {}
    step_data = {} # To store tensors needed for backprop later
    
    # Iterate over current active agents to decide actions
    for agent in env.agents:
        mental_map = env.mental_map[agent]
        mask = env.action_mask_to_node[int(env.agent_position[agent])]
        agent_preds=agent
        # Forward pass
        # obs_nets is a dict of ActorCritic models
        action_logits, value = obs_nets[agent](mental_map=mental_map, mask=mask, agent_preds = agent_preds)
        
        # Sample action
        dist = Categorical(logits=action_logits)
        action = dist.sample()
        
        actions[agent] = action
        
        # Store data for learning phase (Log Prob and Value)
        step_data[agent] = {
            "log_prob": dist.log_prob(action),
            "value": value
        }

    # --- PHASE 2: STEP ENVIRONMENT ---
    # We step ONCE with all actions
    # Note: PettingZoo returns dicts keyed by agent
    obs, rewards, terminations, truncations, infos = env.step(actions)

    # --- PHASE 3: COMPUTE LOSS & BACKPROP ---
    # We iterate over the agents that acted in this step
    for agent, data in step_data.items():
        
        # Get the reward returned by env.step
        reward = rewards.get(agent, 0)
        
        # Determine if agent is done
        done = terminations.get(agent, False) or truncations.get(agent, False)

        # Get Next Value (Bootstrap)
        # We need the value of the NEW state (obs) to calculate the TD target
        # If the agent is dead (done), next value is 0.
        if done:
            next_value = 0
        else:
            # We must run the model again on the new observation (no grad needed)
            with torch.no_grad():
                next_mental_map = env.mental_map[agent] # Ensure this is updated by env.step
                next_mask = env.action_mask_to_node[int(env.agent_position[agent])]
                _, next_value = obs_nets[agent](mental_map=next_mental_map, mask=next_mask)
                next_value = next_value.item() # get scalar

        # Retrieve stored variables from Phase 1
        log_prob = data["log_prob"]
        value = data["value"]
        net_loss=0
        
        # Calculate Loss
        loss = compute_ac_loss(log_prob, value, reward, next_value,done, GAMMA) + net_loss

        # Optimization Step
        optimizers[agent].zero_grad()
        loss.backward()
        optimizers[agent].step()
        

        # Logging
        critic_loss_dict[agent] = loss.item()
        
    # Logging
    logging.info(f"Rewards: {list(rewards.values())}")
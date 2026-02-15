from custom_graph_env import GraphEnv
import torch
import os
from torch.distributions import Categorical
from model import observation_processing_network
import logging
from gymnasium.wrappers import RecordEpisodeStatistics
logger = logging.Logger(f"Logger.log")
handler = logging.Handler(level=0)
attach = logger.addHandler(handler)

def compute_loss(reward, log_probs, value, gamma=0.99):
    # 1. Calculate discounted returns (backwards)
    returns = []
    R = 0
    R = reward + gamma * R
    returns.insert(0, R)
    
    returns = torch.tensor(returns)

    # 2. Calculate Advantage (Actual Return - Predicted Value)
    advantage = returns - value.item()

    # 3. Final Losses
    actor_loss = -(log_probs * advantage).mean()
    critic_loss = torch.nn.functional.mse_loss(value, returns)

    return actor_loss + critic_loss
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
    
    # --- PHASE 1: COLLECT ACTIONS ---
    actions = {}
    step_data = {} # To store tensors needed for backprop later
    
    # Iterate over current active agents to decide actions
    for agent in env.agents:
        mental_map = env.mental_map[agent]
        mask = env.action_mask_to_node[int(env.agent_position[agent])]

        # Forward pass
        # obs_nets is a dict of ActorCritic models
        action_logits, value = obs_nets[agent](mental_map=mental_map, mask=mask)
        
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

        # Calculate Loss
        loss = compute_loss(log_prob, value, reward, next_value, done, GAMMA)

        # Optimization Step
        optimizers[agent].zero_grad()
        loss.backward()
        optimizers[agent].step()
        
        # Logging
        critic_loss_dict[agent] = loss.item()

    # Logging
    logging.info(f"Rewards: {list(rewards.values())}")
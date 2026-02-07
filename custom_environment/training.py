from custom_graph_env import GraphEnv
import torch
from torch.distributions import Categorical
from model import observation_processing_network


def compute_loss(reward, log_probs, value, gamma=0.99):
    # 1. Calculate discounted returns (backwards)
    returns = []
    R = 0
    R = reward + gamma * R
    returns.insert(0, R)
    
    returns = torch.tensor(returns)
    log_probs = torch.stack(log_probs)

    # 2. Calculate Advantage (Actual Return - Predicted Value)
    advantage = returns - value.item()

    # 3. Final Losses
    actor_loss = -(log_probs * advantage).mean()
    critic_loss = torch.nn.functional.mse_loss(value.item(), returns)

    return actor_loss + critic_loss

env = GraphEnv(num_nodes=100)
print(env.agent_position)
observation_processing_network(env.graph.number_of_nodes())
obs_nets:dict = {agent:observation_processing_network(env.graph.number_of_nodes()) for agent in env.possible_agents}
optimizers = {agent:torch.optim.LBFGS(obs_nets[agent].parameters()) for agent in env.agents}
gamma = 0.99
critic_loss_dict:dict = {}
while env.agents :
    actions = {}
    for agent in env.agents:
        action_logits,value = (obs_nets[agent](mental_map = env.mental_map[agent], mask = env.action_mask_to_node[str(env.agent_position[agent])]))
        actions[agent] = Categorical(logits=action_logits).sample()
        log_probs = torch.stack(agent)
        optimizers[agent].zero_grad()

        loss = compute_loss(rewards[agent],action_logits,value)
        
        loss.backward()
        optimizers[agent].step()
    obs, rewards, terminations, truncations, infos = env.step(actions)
       

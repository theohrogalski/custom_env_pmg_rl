from custom_graph_env import GraphEnv
import torch
from torch.distributions import Categorical
from model import observation_processing_network
import logging
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

observation_processing_network(env.graph.number_of_nodes())
obs_nets:dict = {agent:observation_processing_network(env.graph.number_of_nodes()) for agent in env.possible_agents}
optimizers = {agent:torch.optim.Adam(obs_nets[agent].parameters()) for agent in env.agents}
gamma = 0.99

critic_loss_dict:dict = {}
#print("starting")
while env.agents :
    actions,rewards = {},{agent:0 for agent in env.agents}
    for agent in env.agents:
        ##print(env.mental_map[agent])
        #print(f"mental map during trianing is {env.mental_map[agent].number_of_nodes()}")
        assert(env.mental_map[agent].number_of_nodes()==num_nodes)
        action_logits,value = (obs_nets[agent](mental_map = env.mental_map[agent], mask = env.action_mask_to_node[int((env.agent_position[agent]))]))
        actions[agent] = Categorical(logits=action_logits).sample()
        log_probs = (action_logits)
        optimizers[agent].zero_grad()

        loss = compute_loss(rewards[agent],action_logits,value)
        optimizers[agent].zero_grad()
        loss.backward()
        optimizers[agent].step()
    obs, rewards, terminations, truncations, infos = env.step(actions)
    logging.info(str(rewards[agent]) for agent in agent)
    #print("one step")
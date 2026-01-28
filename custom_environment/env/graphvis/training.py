from custom_graph_env import GraphEnv
import torch
from torch.distributions import Categorical
from model import observation_processing_network
env = GraphEnv()
obs_net = observation_processing_network(env.graph.number_of_nodes())
observations, infos = env.reset()

optimizer = torch.optim.Adam(obs_net.parameters())
loss_fn = torch.nn.CrossEntropyLoss()

while env.agents :
    actions = {}
    for agent in env.agents:
        for agent in env.agents:
            action_logits = (obs_net(mental_map = env.mental_map[agent],history = env.mental_map_history[agent], mask = env.action_mask_to_node[f"node_{env.agent_position[agent]}"]))
            actions[agent] = Categorical(logits=action_logits).sample()
    obs, rewards, terminations, truncations, infos = env.step(actions)

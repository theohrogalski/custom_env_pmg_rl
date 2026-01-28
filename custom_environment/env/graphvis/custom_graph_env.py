import asyncio
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import networkx as nx
import random
from torch_geometric.utils import from_networkx
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
from torch import tensor
from stable_baselines3 import DQN, PPO, A2C
import torch
from pettingzoo.utils import aec_to_parallel
import supersuit as ss
from torch.nn.functional import scaled_dot_product_attention
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import VecMonitor
from pettingzoo.test import api_test
import pettingzoo
import functools
import itertools
from pettingzoo.utils.agent_selector import agent_selector
from pettingzoo.utils import wrappers
import time
#print(plt.get_backend())

class GraphEnv(pettingzoo.ParallelEnv):
    def __init__(self, num_nodes=10, num_agents=4, seed=1,render_mode="human", graph_selection=1):
        self.seed=seed
    
        self.render_mode = 2
        self.render_flag = False
        self.metadata = {
        "render_modes": ["human"],  
        "name": "graph_env_v0",
        "is_parallelizable":True
        }
        self.render_mode=render_mode
        self.np_random_seed = int(np.random.randint(1, 10 + 1))
        self.graph:nx.Graph = self.select_graph(load_param=graph_selection,loaded_graphml_name="random_output_graph",)
        self.possible_agents = [f"agent_{k}" for k in range(num_agents)]
        self.total_map_observation = {agent:("") for agent in self.possible_agents}
        self.agent_position={f"agent_{k}":0 for k in self.possible_agents}
        self.agent_name_mapping = dict(
            zip(self.possible_agents, list(range(len(self.possible_agents))))
        )
        
        self.num_nodes = num_nodes
        self.action_spaces = {agent:spaces.Discrete(num_nodes) for agent in self.possible_agents}  # move to node
        self.agents = self.possible_agents
        self._cumulative_rewards = {agent:0 for agent in self.agents}
        self.num_moves = 0
        self.max_uncertainty:int = 100
        self.mmap = {agent:nx.Graph() for agent in self.possible_agents}
        self.mistakes = {agent:0 for agent in self.possible_agents}
        #print(f"node uncertainty is {self.node_unc}")
        self.rewards = {agent:0 for agent in self.agents}
        self.infos = {agent:{} for agent in self.agents}
        self.covered = set()
        self.current_obs = {agent:None for agent in self.possible_agents}
        self.personal_graph = {agent:None for agent in self.possible_agents}
        #self.per_agent_covered = {agent:set() for agent in self.possible_agents}

        self.max_moves=10000
        # Linearly Decaying Parameters
        self.d0= 1
        self.d_k=0

        self.mental_map= {agent:nx.Graph() for agent in self.possible_agents}
        
        self.node_history = {node:[] for node in self.graph.nodes}
        self.neighbors_iter = {node:self.graph.neighbors(node) for node in self.graph}
        self.action_mask_to_node = {node:[0]*self.graph.number_of_nodes() for node in self.graph}
        for node in self.graph.nodes:
            for index in range(self.graph.number_of_nodes()):
                if (index in (self.neighbors_iter[node])) or (index==node[0]):
                    self.action_mask_to_node[node][index] = 1
                    
    def select_graph(self, load_param:int, loaded_graphml_name:str, output_name:str="default_name"):
        if load_param==1:
            return nx.read_graphml(f"{loaded_graphml_name}.graphml")
        elif load_param == 0 :
            self.create_custom_nx_graph(output_name=f"{output_name}")
            return nx.read_graphml(f"{output_name}.graphml")
        else:
            print("empty graph being used\n")
            return nx.Graph()
    def spatial_encoding(self) :
        pass
    def create_custom_nx_graph(self, output_name:str="random_output_graph", num_nodes:int=40, random_chance_param_target:int=20, random_chance_param_edge:float=40) -> None:
        r_graph = nx.Graph()

        for i in range(num_nodes):
            r_graph.add_node((f"{i}"))
            r_graph.nodes[i]["uncertainty"] = 0
            r_graph.nodes[i]["agent_presence"] = 0
            r_graph.nodes[i]["target"] = 0

            random_chance_param_target=min(1,random_chance_param_target)
            random_chance_param_target=max(0,random_chance_param_target)
            random_chance_param_edge=max(0,random_chance_param_edge)
            random_chance_param_edge = min(1,random_chance_param_edge)
            if random.randint(1,100)<random_chance_param_target:
                r_graph.nodes[i]["target"] = 1
        combinations_list = itertools.combinations(r_graph.nodes,2)
        for combo in combinations_list:
            if random.randint(1,100)<random_chance_param_edge*100:
                r_graph.add_edge(combo[0],combo[1])
        nx.write_graphml(r_graph,f"./graphs/{output_name}.graphml")
  
    def reset(self):
        self.node_history = {node:[] for node in self.graph.nodes}
        self.mental_map_history = {agent:[] for agent in self.possible_agents}
        self.mental_map= {agent:nx.Graph() for agent in self.possible_agents}
        self.num_moves=0
        for agent in self.possible_agents:
            self.agent_position[agent] =  int(agent[6:])
            self.rewards[agent] = 0
            self._cumulative_rewards[agent] = 0
            self.infos[agent] = {}
            
        self.agents = (self.possible_agents).copy()
        self._agent_selector = agent_selector(self.agents)
        
        self.terminations = {agent:False for agent in self.agents}
        self.observations = {agent: 0 for agent in self.agents}

        self.truncations = {agent:False for agent in self.agents}
        self.timestep = 0
        # Resetting the graph to have no uncertainty values and accurate edge connections if necessary
        print(self.graph)
        deg = self.graph.degree
        degree_list = [
        ]   
        print(type(deg))
        for item in deg:
            degree_list.append(item)
        for node in range(len(self.graph.nodes)):
            # Resetting uncertainty
            self.graph.nodes[f"node_{node}"]["uncertainty"] = 0
            #Resetting number of connections
            self.graph.nodes[f"node_{node}"]["connections"] = degree_list[node]

        return self.agent_position, {}
    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return gym.spaces.Discrete(self.graph.number_of_nodes(), seed=self.np_random_seed)
    
    @functools.lru_cache(maxsize=None)
    def observation_space(self,agent) :
        return {
            # Observation space for each agent with nodes which have 3 features and 
        agent: gym.spaces.Graph(
            node_space=gym.spaces.Box(low=0, high=1, shape=(3,)), 
            edge_space=gym.spaces.Box(low=0, high=1, shape=(0,))
        ) for agent in self.possible_agents
    }

    def step(self, action:dict):
        rewards, obs, infos = {},{},{}
        for agent in self.agents:
            self.agent_position[agent] = action[agent] 

            for node_idx in range(len(self.graph.nodes)):
                if node_idx in (self.agent_position.values()):
                    self.graph.nodes[f"node_{node_idx}"]["agent_presence"] = 1
                else:
                    self.graph.nodes[f"node_{node_idx}"]["agent_presence"] = -1
                # 
               # self.node_history[agent].append(nodes)
                self.graph.nodes[f"node_{node_idx}"]["uncertainty"] = self.graph.nodes[f"node_{node_idx}"]["agent_presence"]*self.graph.nodes[f"node_{node_idx}"]["target"]

            self.mental_map[agent].add_nodes_from(ego := nx.ego_graph(self.graph,f"node_{self.agent_position[agent]}", radius=2), data=True)
            self.mental_map[agent].add_edges_from(ego.edges(),data=True)
            self.mental_map_history[agent].append((from_networkx(self.mental_map[agent])))
            uncertainty_sum = 0
            for node_index in range(len(self.graph.nodes)):
                uncertainty_sum += self.graph.nodes[f"node_{node_index}"]["uncertainty"]
            rewards[agent] = self.d0*(1-(self.num_moves/self.max_moves))*self.graph.number_of_nodes()-uncertainty_sum
        
        self.truncations = {
            agent: self.num_moves >= self.max_moves for agent in self.agents
            }
        self.num_moves+=1
        
        if self.render_mode == "human":
            self.render()
        
        obs = {agent:{"observation":self.mental_map[agent],"action_mask":self.action_mask_to_node[f"node_{self.agent_position[agent]}"]} for agent in self.agents}
        return obs, rewards, self.terminations, self.truncations, infos
    def observe(self, agent):
        # Every agent sees its mental map
        return self.mental_map[agent]
    def render(self):
        #plt.clf()
        #print(self.agent_position)
        plt.subplot(2,1,1)
        nx.draw_networkx(self.graph)
        
        #print(self.ly)
        if self.num_moves%50==0:
            plt.subplot(2,1,2)
            plt.pause(1)
if __name__ == "__main__":
    env = GraphEnv()
    env.create_custom_nx_graph(output_name="int_name_graph")
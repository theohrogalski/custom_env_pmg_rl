import asyncio
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import networkx as nx
import random
import os
from filterpy.kalman import UnscentedKalmanFilter as ukf
import gpytorch 
import wandb
from torch_geometric.utils import from_networkx
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
from torch import tensor
from random import randint
from stable_baselines3 import DQN, PPO, A2C
import torch
from copy import copy
from pettingzoo.utils import aec_to_parallel
import supersuit as ss
from torch.nn.functional import scaled_dot_product_attention
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import VecMonitor
from pettingzoo.test import api_test
import pettingzoo
import functools
from filterpy.kalman import MerweScaledSigmaPoints
import itertools
from pettingzoo.utils.agent_selector import agent_selector
from pettingzoo.utils import wrappers
import time
from neural_model import uncertainty_estimator as ue
###print(plt.get_backend())

class GraphEnv(pettingzoo.ParallelEnv):
    def __init__(self, num_nodes=50, num_agents=4, seed=1,render_mode="human", graph_selection=0):
        self.seed=seed
        self.num_nodes = num_nodes

        self.render_mode = 2
        self.render_flag = False
        self.metadata = {
        "render_modes": ["human"],  
        "name": "graph_env_v0",
        "is_parallelizable":True
        }
        
        self.render_mode=render_mode
        self.random_num = randint(0,10000)
        self.np_random_seed = int(np.random.randint(1, 10 + 1))
        self.graph:nx.Graph = self.select_graph(load_param=1,output_name=f"graph_{self.random_num}",loaded_graphml_name="./graphs/graph_8114")
        ##print(f"Before: {list(self.graph.nodes)} | Type: {type(list(self.graph.nodes)[0])}")
        
        # 2. Define your type conversion (e.g., str -> int)
        # We create a mapping: {old_id: new_id}
        mapping = {node: int(node) for node in self.graph.nodes()}
        
        # 3. Modify the graph in-place
        # copy=False ensures the original G is modified
        nx.relabel_nodes(self.graph, mapping, copy=False)
        ##print(f"After:  {list(self.graph.nodes)} | Type: {type(list(self.graph.nodes)[0])}")
        """for i in self.graph.nodes:
            ##print(self.graph.nodes[i]["uncertainty"]) """
        self.possible_agents = [f"agent_{k}" for k in range(num_agents)]
        self.total_map_observation = {agent:("") for agent in self.possible_agents}
        self.agent_position={agent:0 for agent in self.possible_agents}
        self.agent_name_mapping = dict(
            zip(self.possible_agents, list(range(len(self.possible_agents))))
        )
        
        self.action_spaces = {agent:spaces.Discrete(self.num_nodes) for agent in self.possible_agents}  # move to node
        self.agents = self.possible_agents
        self._cumulative_rewards = {agent:0 for agent in self.agents}
        self.num_moves = 0
        self.obs_dict = {node:torch.Tensor() for node in range(self.num_nodes)}

        self.model_path = "./saved_models"
        self.max_uncertainty:int = 100
        self.mmap = {agent:nx.Graph() for agent in self.possible_agents}
        self.mistakes = {agent:0 for agent in self.possible_agents}
        ###print(f"node uncertainty is {self.node_unc}")
        self.rewards = {agent:0 for agent in self.agents}
        self.infos = {agent:{} for agent in self.agents}
        self.covered = set()
        self.current_obs = {agent:None for agent in self.possible_agents}
        self.personal_graph = {agent:None for agent in self.possible_agents}
        #self.per_agent_covered = {agent:set() for agent in self.possible_agents}
        self.terminations = {agent:False for agent in self.agents}

        self.max_moves=10000
        # Linearly Decaying Parameters
        self.d0= 1
        self.d_k=0
        starting_graph:nx.Graph = nx.Graph()
        for n in range(self.num_nodes):
            starting_graph.add_node(int(n))
        for node in starting_graph.nodes():
            starting_graph.nodes[node]["uncertainty"]=0
            starting_graph.nodes[node]["agent_presence"]=0
            starting_graph.nodes[node]["target"]=0
        ##print(f" starting graph is {starting_graph.nodes()}")

        self.mental_map= {agent:starting_graph.copy() for agent in self.possible_agents}
        ##print(self.mental_map["agent_0"])
      #  ##print(f"start nodes is {starting_graph.number_of_nodes()}")
        for agent in self.possible_agents:   
            ego = nx.ego_graph(self.graph, int(self.agent_position[agent]), radius=2)

            ##print(f"before {self.mental_map[agent].number_of_nodes()}")
            
            self.mental_map[agent].add_nodes_from(ego.nodes(data=True))
            ##print(list(ego.nodes()))
            ##print(list(self.mental_map[agent].nodes()))
            ##print("added nodes")
            self.mental_map[agent].add_edges_from(ego.edges(data=True))
        self.episode_num=0
        self.reward_graph={agent:[] for agent in self.possible_agents}
        self.total_uncertainty_graph=[]
        self.node_history = {node:[] for node in self.graph.nodes}
        self.neighbors_iter = {node:list(self.graph.neighbors(node))for node in self.graph.nodes}
        self.action_mask_to_node = {node:[0]*(num_nodes) for node in self.graph}

        """for node in self.graph.nodes:
            ##print(len(list(self.graph.neighbors(n=node))))"""


        for node in self.graph.nodes:
            for index in range(self.graph.number_of_nodes()):
                ###print(index)
                ###print(self.neighbors_iter[node])
                if ((index) in (self.neighbors_iter[node])) or index==int(node) :
                    
                    self.action_mask_to_node[node][index] = 1
        ###print(self.action_mask_to_node)   
        uncertainty_estimator = ue()  
        node_to_net = {node:copy(uncertainty_estimator) for node in range(self.num_nodes)}
        self.agent_to_net = {agent:copy(node_to_net) for agent in self.possible_agents}

        self.predictions_to_agent = {agent:[] for agent in self.possible_agents}
    def select_graph(self, load_param:int, loaded_graphml_name:str, output_name:str="default_name",):
        """
        Docstring for select_graph
        
        :param load_param: 1 for loading a graph from a graphml file, 0 for using the create custom nx graph method
        :type load_param: int
        :param loaded_graphml_name: Name of the desired loaded graphml file, eg "example" <-> example.graphml
        :type loaded_graphml_name: str
        :param output_name: The output name of the created graph if load_param is 0
        :type output_name: str
        """
        if load_param==1:
            return nx.read_graphml(f"{loaded_graphml_name}.graphml")
        elif load_param == 0 :
            self.create_custom_nx_graph(output_name=f"{output_name}")
            
            return nx.read_graphml(f"./graphs/{output_name}.graphml")
        else:
            ##print("empty graph being used\n")
            return nx.Graph()
    def len_path_in_mm(self,agent)->list:
        shortest_path_list=[]
        for node in self.mental_map[agent].nodes():
            if self.graph.nodes[node]["agent_presence"]==1:
                shortest_path_list.append(nx.shortest_path_length(self.graph,self.agent_position[agent],node))
        return(shortest_path_list)
    def create_custom_nx_graph(self, output_name:str="random_output_graph", random_chance_param_target:int=20, random_chance_param_edge:int=20) -> None:
        r_graph = nx.Graph()
        for i in range(self.num_nodes):
            r_graph.add_node(int(i))

            r_graph.nodes[i]["uncertainty"] = 0
            r_graph.nodes[i]["agent_presence"] = 0
            r_graph.nodes[i]["target"] = 0
            r_graph.nodes[i]["coordinates"] = (randint(0,self.num_nodes*3),randint(0,self.num_nodes*3))
            
            if random.randint(1,100)<random_chance_param_target:
                r_graph.nodes[i]["target"] = 1
        combinations_list = itertools.combinations(r_graph.nodes,2)
        for combo in combinations_list:
            if random.randint(1,100)<random_chance_param_edge:
                r_graph.add_edge(combo[0],combo[1])
        nx.write_graphml(r_graph,f"./graphs/{output_name}.graphml")
        ##print("got here")
  
    def reset(self):
        print(len(self.total_uncertainty_graph))
        plt.plot(self.total_uncertainty_graph)
        plt.title(f"Uncertainty_Graph_{self.episode_num}")
        plt.show()
        plt.savefig("uncertainty_graph.png")
        self.episode_num +=1
        
        self.obs_dict = {node:torch.Tensor() for node in range(self.num_nodes)}

        ##print("running reset")
        self.graph:nx.Graph = self.select_graph(load_param=1,output_name=f"graph_{self.random_num}",loaded_graphml_name="./graphs/graph_8114")
            
        assert(type(self.graph.nodes()[0])==int)
        self.node_history = {node:[] for node in self.graph.nodes}
        starting_graph:nx.Graph = nx.Graph()
        for n in range(self.num_nodes):
            starting_graph.add_node(int(n))
        for node in starting_graph.nodes():
            starting_graph.nodes[node]["uncertainty"]=0
            starting_graph.nodes[node]["agent_presence"]=0
            starting_graph.nodes[node]["target"]=0
        ##print(f"nodes from reset is {starting_graph.number_of_nodes()}")
        assert(starting_graph.number_of_nodes()==self.num_nodes)
        self.num_moves=0
        for agent in self.possible_agents:
            self.agent_position[agent] =  int(agent[6:])
            self.rewards[agent] = 0
            self._cumulative_rewards[agent] = 0
            self.infos[agent] = {}
        self.mental_map= {agent:starting_graph.copy() for agent in self.possible_agents}
        for agent in self.possible_agents:   
            ego = nx.ego_graph(self.graph, int(self.agent_position[agent]), radius=2)

            ##print(f"before {self.mental_map[agent].number_of_nodes()}")
            
            self.mental_map[agent].add_nodes_from(ego.nodes(data=True))
            ##print(list(ego.nodes()))
            ##print(list(self.mental_map[agent].nodes()))
            ##print("added nodes")
            self.mental_map[agent].add_edges_from(ego.edges(data=True))

        self.reward_graph={agent:[] for agent in self.possible_agents}
        self.agents = (self.possible_agents).copy()
        self._agent_selector = agent_selector(self.agents)
        
        self.terminations = {agent:False for agent in self.agents}
        self.observations = {agent: 0 for agent in self.agents}

        self.truncations = {agent:False for agent in self.agents}
        self.timestep = 0
        # Resetting the graph to have no uncertainty values and accurate edge connections if necessary
        deg = self.graph.degree
        degree_list = [
        ]   
        ##print(type(deg))
        for item in deg:
            degree_list.append(item)
        for node in range(len(self.graph.nodes())):
            # Resetting uncertainty
            self.graph.nodes[node]["uncertainty"] = 0
            #Resetting number of connections
            self.graph.nodes[node]["connections"] = degree_list[node]
        self.agent_position={agent:0 for agent in self.possible_agents}
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
        values = [data["uncertainty"] for node, data in self.graph.nodes(data=True)]
        #print(values)
        #print(self.agent_position.values())
        # Reset rewards, observations, infos
        assert(len(self.graph.nodes())>0)
        ##print(f" here2 {self.graph.nodes()}")
        rewards, obs, infos = {},{},{}
        # Loop through
        for agent in self.agents:
            self.agent_position[agent] = action[agent].item()

            for node_idx in range(len(self.graph.nodes())):
                
                if node_idx in (self.agent_position.values()):
                    ##print(f"here {list(self.graph.nodes())}")
                    self.graph.nodes[(node_idx)]["agent_presence"] = 1
                else:
                    self.graph.nodes[(node_idx)]["agent_presence"] = 0
                # 
               # self.node_history[agent].append(nodes)
                if self.graph.nodes[node_idx]["target"]==1 and self.graph.nodes[(node_idx)]["agent_presence"] == 1 and self.graph.nodes[(node_idx)]["uncertainty"]>0:

                    self.graph.nodes[(node_idx)]["uncertainty"]-=1
                    ##print("minused uncertainty")
                elif self.graph.nodes[(node_idx)]["uncertainty"]<self.max_uncertainty and self.graph.nodes[node_idx]["target"]==1:
                    self.graph.nodes[(node_idx)]["uncertainty"]+=1
                    
               
            ego = nx.ego_graph(self.graph, int(self.agent_position[agent]), radius=2)
            ego_nodes_list:list = [ego.nodes()]

            for node in range(self.num_nodes):
                if node in ego_nodes_list:
                    self.agent_to_net[agent][node].data.append(
                        self.graph.nodes[node]["uncertainty"]
                                    )
            pred_list=[]

            for node in range(self.num_nodes):
                pred_list=[]
                pred_list.append(self.agent_to_filter[agent][node].forward())
            self.predictions_to_agent[agent]=pred_list

                
            ##print(f"before {self.mental_map[agent].number_of_nodes()}")
            
            self.mental_map[agent].add_nodes_from(ego.nodes(data=True))
            ##print(list(ego.nodes()))
            ##print(list(self.mental_map[agent].nodes()))
            ##print("added nodes")
            self.mental_map[agent].add_edges_from(ego.edges(data=True))
            for node in self.mental_map[agent].nodes():
                self.obs_dict[node] = torch.cat( (self.obs_dict[node], torch.Tensor(int(self.graph.nodes[node]["uncertainty"])) ))
            ##print(self.mental_map[agent].number_of_nodes())

            uncertainty_sum=0
            for node_index in range(len(self.graph.nodes)):
                uncertainty_sum += self.graph.nodes[(node_index)]["uncertainty"]
            self.total_uncertainty_graph.append(uncertainty_sum)
            rewards[agent] = self.d0*(1-(self.num_moves/self.max_moves))*self.graph.number_of_nodes()-uncertainty_sum*0.1
            ##print(rewards[agent])
        self.truncations = {
            agent: self.num_moves >= self.max_moves for agent in self.agents
            }
        self.num_moves+=1
        
        if self.render_mode == "human":
            self.render()
        
        obs = {agent:{"observation":self.mental_map[agent],"action_mask":self.action_mask_to_node[(self.agent_position[agent])]} for agent in self.agents}


        return obs, rewards, self.terminations, self.truncations, infos
    def observe(self, agent):
        # Every agent sees its mental map
        return self.mental_map[agent]
    def render(self):
       pass
        #plt.clf()
        ###print(self.agent_position)
        #plt.subplot(2,1,1)
        #nx.draw_networkx(self.graph)
        
        ###print(self.ly)
       """ if self.num_moves%50==0:
            plt.subplot(2,1,2)
            plt.pause(1)"""
if __name__ == "__main__":
    env = GraphEnv()
    env.create_custom_nx_graph(output_name="int_name_graph")
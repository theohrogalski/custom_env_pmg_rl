import asyncio
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import networkx as nx
import random
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
from stable_baselines3 import DQN, PPO, A2C
from pettingzoo.utils import aec_to_parallel
import supersuit as ss
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import VecMonitor
from pettingzoo.test import api_test
import pettingzoo
from stable_baselines3.common.evaluation import evaluate_policy
import functools
import itertools
from pettingzoo.utils.agent_selector import agent_selector
from pettingzoo.utils import wrappers
import time
#print(plt.get_backend())

class GraphEnv(pettingzoo.ParallelEnv):
    def __init__(self, num_nodes=10, num_agents=2, seed=1,render_mode="human", graph_selection=1):
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
        self.graph:nx.Graph = self.select_graph(load_param=graph_selection,loaded_graphml_name="random_graph_ml",)
        self.possible_agents = [f"agent_{k}" for k in range(num_agents)]
        self.total_map_observation = {agent:("") for agent in self.possible_agents}
        self.agent_position={f"agent_{k}":"node_null" for k in self.possible_agents}
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

        self.mental_map= {agent:nx.Graph() for agent in self.possible_agents}

    def select_graph(self, load_param:int, loaded_graphml_name:str, output_name:str="default_name"):
        if load_param==1:
            return nx.read_graphml(f"{loaded_graphml_name}.graphml")
        elif load_param == 0 :
            self.create_custom_nx_graph(output_name=f"{output_name}")
            return nx.read_graphml(f"{output_name}.graphml")
        else:
            print("empty graph being used\n")
            return nx.Graph()
        
    def create_custom_nx_graph(self, output_name:str="random_output_graph", num_nodes:int=40, random_chance_param_target:float=0.2, random_chance_param_edge:float=0.4) -> None:
        r_graph = nx.Graph()

        for i in range(num_nodes):
            r_graph.add_node((f"node_{i}"))
            r_graph.nodes[f"node_{i}"]["uncertainty"] = 0
            r_graph.nodes[f"node_{i}"]["agent_presence"] = 0
            r_graph.nodes[f"node_{i}"]["target"] = 0

            random_chance_param_target=min(1,random_chance_param_target)
            random_chance_param_target=max(0,random_chance_param_target)
            random_chance_param_edge=max(0,random_chance_param_edge)
            random_chance_param_edge = min(1,random_chance_param_edge)
            if random.randint(1,100)<random_chance_param_target*100:
                r_graph.nodes[f"node_{i}"]["target"] = 1
        combinations_list = itertools.combinations(r_graph.nodes,2)
        for combo in combinations_list:
            if random.randint(1,100)<random_chance_param_edge*100:
                r_graph.add_edge(combo[0],combo[1])
        nx.write_graphml(r_graph,f"{output_name}.graphml")

   

    def reset(self, options=None,seed=None):
        self.covered = set()
        self.num_moves=0
        for agent in self.possible_agents:
            self.agent_position[agent] = f"node_{agent[5:]}"
            self.rewards[agent] = 0
            self._cumulative_rewards[agent] = 0
            self.infos[agent] = {}
            
        self.agents = (self.possible_agents).copy()
        self._agent_selector = agent_selector(self.agents)
        
        if self._agent_selector:
            self.agent_selection = self._agent_selector.next()
        self.terminations = {agent:False for agent in self.agents}
        self.observations = {agent: 0 for agent in self.agents}

        self.truncations = {agent:False for agent in self.agents}
        self.timestep = 0

        return self.agent_position, {}
    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        # We can seed the action space to make the environment deterministic.
        return gym.spaces.Discrete(self.graph.number_of_nodes(), seed=self.np_random_seed)
    @functools.lru_cache(maxsize=None)
    def observation_space(self,agent) :
        return gym.spaces.Graph(gym.spaces.Discrete(1),None)

    def step(self, action):
        self.agent_position[self.agent_selection] = action
        
        for nodes in self.graph.nodes:
            if nodes in self.agent_position.items():
                nodes["agent_presence"] = 1
            else:
                nodes["agent_presence"] = 0
            nodes["uncertainty"] += nodes["target"]*nodes["agent_presence"]

        self.mental_map[self.agent_selection].add_nodes_from(ego := nx.ego_graph(self.graph,self.agent_position[self.agent_selection], radius=2))
        self.mental_map[self.agent_selection].add_edges_from(ego)
        
        
        if (
            self.terminations[self.agent_selection]
            or self.truncations[self.agent_selection]
        ):
            # the next dead agent,  or if there are no more dead agents, to the next live agent
            self._was_dead_step(action)
            return

        

        #(self.covered).add(self.agent_position[self.agent_selection])
        if self._agent_selector.is_last():
            self.num_moves += 1
            
            # The truncations dictionary must be updated for all players.
            self.truncations = {
                agent: self.num_moves >= 1000 for agent in self.agents
            }

        else:
            self._clear_rewards()
        # selects the next agent.
        self.agent_selection = self._agent_selector.next()
        # Adds .rewards to ._cumulative_rewards
        
        self._accumulate_rewards()
        if self.render_mode == "human":
            self.render()
    def observe(self, agent):
    # simplest: every agent just sees the global state
        return self.mental_map[agent]
    def render(self, total_reward=None):
        #plt.clf()
        #print(self.agent_position)
        plt.subplot(2,1,1)
        nx.draw_networkx(self.graph)
        
        #print(self.ly)
        if self.num_moves%50==0:
            plt.subplot(2,1,2)
            plt.pause(1)
        #mngr = plt.get_current_fig_manager()
        #mngr.window.wm_geometry((f"500x500+100+100"))        # TkAgg backend (most common)

        #plt.ion()
        # to put it into the upper left corner for example:
        #plt.pause(0.75)

        #plt.close()
        
        
        
    

env=GraphEnv(num_nodes = 40, num_agents=6)
graph = env.create_custom_nx_graph(num_nodes=40)

vec_env = ss.pettingzoo_env_to_vec_env_v1(env)
vec_env = ss.concat_vec_envs_v1(vec_env, 1, num_cpus=8, base_class="stable_baselines3")
vec_env = VecMonitor(vec_env,filename="./log_dir")
"""model = DQN("MlpPolicy", vec_env, verbose=1)
timestart = time.time()
model.learn(total_timesteps=10,progress_bar=True)
print(f"total time = {time.time()-timestart}")
model.save(f"policy_")"""
#mean_reward, std_reward = evaluate_policy(model=model, env=vec_env)
#print(f"mean reward for DQN is {mean_reward}, std_reward for DQN is {std_reward}")
model_dqn = DQN("MlpPolicy", vec_env, verbose=1)
#model_ppo = PPO("MlpPolicy", vec_env, verbose =1)
mean_reward_dqn,std_reward_dqn= evaluate_policy(model=model_dqn, env=vec_env)
print(f"mean reward without any learning{mean_reward_dqn}")
model_dqn.learn(total_timesteps=300,progress_bar=True)
#model_ppo.learn(total_timesteps=100, progress_bar=True)
#model_dqn.save("dqn_model_savefile")


#model_ppo.save("dqn_save")
mean_reward_dqn,std_reward_dqn= evaluate_policy(model=model_dqn, env=vec_env)
#mean_reward_ppo, std_reward_ppo = evaluate_policy(model=model_ppo,env=vec_env)
print(f"mean reward for dqn is {mean_reward_dqn}")
#print(f"mean reward for dqn is {mean_reward_ppo}")

print("With num training steps being 100")
model_dqn.load("./dqn_model_savefile.zip")
mean_reward_dqn,std_reward_dqn= evaluate_policy(model=model_dqn, env=vec_env)
print(mean_reward_dqn)
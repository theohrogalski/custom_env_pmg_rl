import asyncio
import gymnasium
import numpy as np
import networkx as nx
import time
import signal
import torch
import random
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
from stable_baselines3 import DQN, PPO, A2C
from pettingzoo.utils import aec_to_parallel
import supersuit as ss
import stable_baselines3
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import VecMonitor
from pettingzoo.test import api_test
import pettingzoo
from stable_baselines3.common.evaluation import evaluate_policy
import functools
from pettingzoo.utils.agent_selector import agent_selector
from pettingzoo.utils import wrappers
import time
#print(plt.get_backend())

class GraphEnv(pettingzoo.AECEnv):
    def __init__(self, num_nodes=10, num_agents=2, seed=1,render_mode="human"):
        self.seed=seed
       
        self.render_mode = 2
        self.render_flag = False
        self.possible_agents = [f"agent_{k}" for k in range(num_agents)]

        self.ly=[]
        # Fully decentralized decision making
        agent_policies = {
            agent: observation_network(obs_dim, action_dim)
            for agent in self.possible
                         }
        # Fully decentralized decision making
        agent_opts = {
        agent: torch.optim.Adam(agent_policies[agent].parameters(), lr=3e-4)
        for agent in env.agents
                     }
        self.metadata = {
        "render_modes": ["human"],   # or ["human"] if you want GUI rendering
        "name": "graph_env_v0",
        "is_parallelizable":True
        }
        self.render_mode=render_mode
        self.np_random_seed = int(np.random.randint(1, 10 + 1))
        self.total_map_observation = {agent:("") for agent in self.possible_agents}
        self.agent_name_mapping = dict(
            zip(self.possible_agents, list(range(len(self.possible_agents))))
        )
        # Graph Definition
        self.graph=nx.Graph()
        nodelist=[ ]
        for i in range(num_nodes):
            nodelist.append(({i:0}))
        self.graph.add_edges_from()

        # Target Definition
        self.targets = [1,2,3,4,5,6,6,7,8,9]


        self.agent_obs_map = {nx.Graph:agent for agent in self.possible_agents}
        self.num_nodes = num_nodes
        self.action_spaces = {agent:spaces.Discrete(num_nodes) for agent in self.possible_agents}  # move to node
        self.observation_spaces = {agent:gymnasium..Discrete(num_nodes) for agent in self.possible_agents}
        self.agents = self.possible_agents
        self._cumulative_rewards = {agent:0 for agent in self.agents}
        self.current_node = {agent: None for agent in self.agents}
        self.num_moves = 0
        self.max_uncertainty:int = 100
        
        self.node_unc = {node:0 for node in self.graph}
        #print(f"node uncertainty is {self.node_unc}")
        self.rewards = {agent:0 for agent in self.agents}
        self.infos = {agent:{} for agent in self.agents}
        self.covered = set()
        #self.per_agent_covered = {agent:set() for agent in self.possible_agents}
        self.mental_map= {agent:nx.Graph() for agent in self.possible_agents}
        self.agent_position = {agent:0 for agent in self.possible_agents}
    def reset(self, options=None,seed=None):
        self.covered = set()
        self.num_moves=0
        self.node_unc = {node:0 for node in self.graph}
        for agent in self.possible_agents:

            self.infos[agent] = {}

        self.agents = (self.possible_agents).copy()
        
        self.terminations = {agent:False for agent in self.agents}
        self.observations = {agent: nx.Graph for agent in self.agents}

        self.truncations = {agent:False for agent in self.agents}
        self.timestep = 0
        return self.current_node, {}
    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        # We can seed the action space to make the environment deterministic.
        return gym.spaces.Discrete(3, seed=self.np_random_seed)
    @functools.lru_cache(maxsize=None)
    def observation_space(self,agent) :
        return gym.spaces.Discrete(self.num_nodes)

    def step(self, action):
        print(list(self.current_node.values()))
        self.agen
       
        self.truncations = {
                agent: self.num_moves >= 1000 for agent in self.agents
            }
            
            # observe the current current_node
        
    def observe(self, agent):
        return nx.Graph(nx.neighbors(self.mental_map[agent],self.current_node[agent]))
    def render(self, total_reward=None):
        #plt.clf()
        #print(self.current_node)
        plt.subplot(2,1,1)
        nx.draw_networkx(self.graph, with_labels=True,pos=nx.spring_layout(self.graph,seed=0),
                node_color=
                [(min((list(self.current_node.values()).count(i)*100)/self.num_agents,0.99),
                  min(1, self.node_unc[i]/self.max_uncertainty),
                  min(1, self.node_unc[i]/self.max_uncertainty)
                  ) for i in self.graph.nodes()]
                  
                  )
        
        #print(self.ly)
        if self.num_moves%50==0:
            plt.subplot(2,1,2)
            plt.plot(self.lx,self.ly)
            plt.pause(1)
        #mngr = plt.get_current_fig_manager()
        #mngr.window.wm_geometry((f"500x500+100+100"))        # TkAgg backend (most common)

        #plt.ion()
        # to put it into the upper left corner for example:
        #plt.pause(0.75)

        #plt.close()
        
        
        
    
        
env=GraphEnv(num_nodes = 40, num_agents=6)

env = wrappers.OrderEnforcingWrapper(env)        
parallel_env = aec_to_parallel(env)

vec_env = ss.pettingzoo_env_to_vec_env_v1(parallel_env)
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
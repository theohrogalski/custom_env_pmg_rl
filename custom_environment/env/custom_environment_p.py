from pettingzoo.utils import parallel_to_aec
from pettingzoo.utils.env import ParallelEnv
import numpy as np
from gymnasium import spaces
import gymnasium
from pettingzoo.test import parallel_api_test
from logging import Logger, FileHandler
from math import ceil
class GridWithMemory(ParallelEnv):

    """
    Parallel envrionment for a grid world where agents can learn and update their map of the environment.

    """
    metadata = { "name": "pmg_env","render_modes": ["human"],}
    def __init__(self, n_agents=3, max_steps=100, num_targets=6, view_size=2, seed=None, grid_size=(10,10),render_mode=None):
        self.grid_size = grid_size 
        
        self.render_mode = render_mode or "human"
        self.max_steps = max_steps
        self.possible_agents = [f"agent_{i}" for i in range(n_agents)]
        self.target_names = [f"target_{j}" for j in range(num_targets)]
        #print(f"self possible agents 0 = {self.possible_agents[0]}")
        self.action_spaces = {
            name: spaces.Discrete(5) # up right left down stay
            for name in self.possible_agents
                            }
        empty_grid = np.full(shape=self.grid_size +(2,),fill_value=-1,dtype=np.float16)
        self.agentwise_grid = {f"agent_{agent}":empty_grid
                               for agent in range(n_agents)}
        # The observation space is defined 
        self.observation_spaces = {
                name: {
                "grid_state": spaces.Box(
                    low=0,
                    high=ceil(grid_size[0] * grid_size[1] * 0.5),
                    shape=(2,),
                    dtype=np.float32
                ),
                "agent_position": spaces.Box(
                    low=0,
                    high=grid_size[0] - 1,
                    shape=(2,),  # assuming (x, y)
                    dtype=np.int32
                    ),
                    }
    for name in self.possible_agents
                                }
        self.rewards = {agent:0
                    for agent in self.possible_agents
                        }
            # These uncertainties are a tuple of possible values. When observed,
            # they enter a state of the lo and hi values being equal and for each timestep the lo and high change by 1 (-,+).
            # Should be formatted according to:
            # distance: (lo,hi)
            
            # The goal of the agent should be to formulate a very strong undertsanding of the map state in order to make hihgly effective 
            # actions. 
            
        self.agentPositions = {}
        self.tPositions = {}
        self.uncertainty = {}
        self.current_step = None
        self.np_random = np.random.default_rng(seed)

    def reset(self, seed=None, options=None):
        self.agents = self.possible_agents.copy()

        self.grid_state = np.full(self.grid_size,fill_value=0.5,dtype=np.float16)
       # print(self.grid_state)
        #print(self.grid_state)
        #[-1,-1] means unknown, [0.5,0.5] means empty
        self.current_step = 0
        if seed is not None:
            self.np_random = np.random.default_rng(seed)
            self.current_step = 0

        # spawn positions
        for name in self.possible_agents:
            self.agentPositions[name] = self.np_random.integers(
                low=[0,0], high=self.grid_size, size=(2,),
            )
        
        self.targets = self.target_names.copy()
        for target in self.targets:
            #if you want a custom target setup plug in here
            #This is of format [x,y]
            self.tPositions[target] = self.np_random.integers(
                low=[0,0], high=self.grid_size, size = (2,)
            )
            
            self.grid_state[tuple(self.tPositions[target])] = 0
            
            #print(self.grid_state)
            # This sets the dynamic uncertainty update up
       # print(self.grid_state)    
        # setting the uncertainty value
        for target in self.target_names:
            self.uncertainty[target] = 0
        obs   = { name: 0     for name in self.agents }
        infos = { name: {}    for name in self.agents }

        return obs, infos

    def step(self, actions):
        self.current_step += 1
        #print(f"self2 possible agents 0 = {self.possible_agents[0]}")
        self.total_unc = sum(self.uncertainty.values())
        terminations = {}
        truncations = {}
        infos = {}

        # 1) Process each agent
        for name, act in actions.items():
            #old_memory = self.agentMemory[name]

            # --- move agent ---
            x, y = self.agentPositions[name]
            if act == 0:    y = min(y + 1, self.grid_size[1] - 1) #up
            
            elif act == 1:  y = max(y - 1, 0) #down
            elif act == 2:  x = max(x - 1, 0) #left 
            elif act == 3:  x = min(x + 1, self.grid_size[0] - 1) #right
            elif act == 4: pass
            # --- compute view & uncertainty-based reward ---
            radius = 3
            low_x, high_x = max(0, x-radius), min(self.grid_size[0], x+radius)
            low_y, high_y = max(0, y-radius), min(self.grid_size[1], y+radius)
            in_view_x = np.arange(low_x, high_x)
            in_view_y = np.arange(low_y, high_y)
            for i in in_view_x:
                for j in in_view_y:
                    current_pos = (i, j)
                    observed_value= self.grid_state[i, j]

            # --- update each agent's internal map ---    
                    # internal map  # current agent  # the ith value [0.5,0.5 if empty]
                    if np.array_equal(self.agentwise_grid[name][i, j], [-1, -1]):
                        self.agentwise_grid[name][i, j] = [observed_value,observed_value]
                        #print(self.agentwise_grid[name])
                    for agent in self.possible_agents:
                        if self.agentPositions[agent][0] in in_view_x and  self.agentPositions[agent][1] in in_view_y:
                            self.agentwise_grid[name][self.agentPositions[agent][0]][self.agentPositions[agent][1]] += 0.01
                            # --- update the uncertainty value for items not in range ---
            in_view = set(zip(in_view_x, in_view_y))
            for x in range(len(self.grid_size)):        
                for y in range(len(self.grid_size)):
                    if (self.agentwise_grid[name][x, y][0] != -1 and
                         self.agentwise_grid[name][x, y][1] != 0.5 and
                             (x, y) not in in_view
                        ):
                        #This range is the distribution of possible uncertainty values at this point. 
                        new_pos = np.add([x, y], [-1, 1])	
                        self.agentwise_grid[name][tuple(new_pos)]

                        if self.agentwise_grid[name][x, y][0] < 0:
                            self.agentwise_grid[name][x, y][0] = 0
            
                
                    # The reward needs to prioritize the local values 
                    # while taking into account the uncertainty ranges at the points in its learned map
                    # as well as the distance from each. 
                    # For example, it should prioritize a wide ranging value versus a low uncertainty one near it
                # --- flags for this agent ---
            terminations[name] = False
            truncations[name]  = True
            if (self.current_step >= self.max_steps):
                if self.total_unc >68:
                    self.rewards[name] = -self.total_unc
                    # WIP For target in known grid, sum the uncertainty 
                    # ALSO ADD IN THE AGENT POSITIONS 
                else:
                    self.rewards[name] = 1

                truncations[name]  = True

           
            infos[name] = self.agentPositions[name]
            obs = {
                    name: {
                    "grid_state": self.agentwise_grid[name],
                    "agent_position": self.agentPositions[name]
                    }
                    for name in self.agents
                        }
            
        # 2) Update all target uncertainties
        for t in self.target_names:
            covered = any(
                (pos == self.tPositions[t]).all()
                for pos in self.agentPositions.values()
            )
            
            delta = -1 if covered else +1
            self.uncertainty[t] = max(0, self.uncertainty[t] + delta)
            #print(self.tPositions[t])
            self.grid_state[tuple(self.tPositions[t])] = self.uncertainty[t]
        # 3) **Prune** any agent you flagged as done/truncated
        #print(self.agentwise_grid[name])
        
        #print(f"post loop: {self.grid_state}")
        self.agents = [
            a for a in self.agents
            if not (terminations.get(a, False) or truncations.get(a, False))
        ]
        
    # 4) Set the global “episode over” flag
        terminations["__all__"] = False
        truncations["__all__"]  = len(self.agents) == 0
        return obs, terminations, self.rewards,truncations, infos

    def render(self):
        print(self.grid_state)
        
    def close(self):
        pass
    def observation_space(self, agent_id):
        return self.observation_spaces[agent_id]
    def action_space(self, agent):
        return self.action_spaces[agent]
    def final_cost(self):
        print(self.rewards)
 
env = GridWithMemory()
parallel_api_test(env)
from pettingzoo.utils import parallel_to_aec
from pettingzoo.utils.env import ParallelEnv
import numpy as np
from gymnasium import spaces
from pettingzoo.test import parallel_api_test

class SimpleGridWorld(ParallelEnv):
    metadata = { "name": "pmg_env","render_modes": ["human"],}
    def __init__(self, n_agents=1, max_steps=100, num_targets=6, view_size=2, seed=None, grid_size=(10,10),render_mode=None):
        self.grid_size = grid_size 
        self.render_mode = render_mode or "human"
        self.max_steps = max_steps
        self.possible_agents = [f"agent_{i}" for i in range(n_agents)]
        self.target_names = [f"target_{j}" for j in range(num_targets)]
        self.agentMemory = {}
        
        self.observation_spaces = {
            name: spaces.Discrete(100) # max uncertainty
            for name in self.possible_agents
        }
        self.action_spaces = {
            name: spaces.Discrete(5)
            for name in self.possible_agents
        }        
    
        self.agentPositions = {}
        self.tPositions = {}
        self.uncertainty = {}
        self.current_step = None
        self.np_random = np.random.default_rng(seed)

    def reset(self, seed=None, options=None):
        self.agents = self.possible_agents.copy()
        self.current_step = 0
        if seed is not None:
            self.np_random = np.random.default_rng(seed)
            self.current_step = 0

        # spawn positions
        for name in self.possible_agents:
            self.agentPositions[name] = self.np_random.integers(
                low=[0,0], high=self.grid_size, size=(2,)
            )
        self.targets = self.target_names.copy()
        for target in self.targets:
            #if you want a custom target setup plug in here
            self.tPositions[target] = self.np_random.integers(
                low=[0,0], high=self.grid_size, size = (2,)
            )
        # setting the uncertainty value
        for target in self.target_names:
            self.uncertainty[target] = 0
        obs   = { name: 0     for name in self.agents }
        infos = { name: {}    for name in self.agents }
        return obs, infos

    def step(self, actions):
        self.current_step += 1

        obs = {}
        rewards = {}
        terminations = {}
        truncations = {}
        infos = {}

        # 1) Process each agent
        for name, act in actions.items():
            # --- move agent ---
            x, y = self.agentPositions[name]
            if act == 0:    y = min(y + 1, self.grid_size[1] - 1) #up
            
            elif act == 1:  y = max(y - 1, 0) #down
            elif act == 2:  x = max(x - 1, 0) #left 
            elif act == 3:  x = min(x + 1, self.grid_size[0] - 1) #right
            # act == 4: stay
            self.agentPositions[name] = np.array([x, y])

            # --- compute view & uncertainty-based reward ---
            radius = 3
            low_x, high_x = max(0, x-radius), min(self.grid_size[0], x+radius+1)
            low_y, high_y = max(0, y-radius), min(self.grid_size[1], y+radius+1)
            in_view = [
                t for t in self.target_names
                if low_x <= self.tPositions[t][0] < high_x
                and low_y <= self.tPositions[t][1] < high_y
            ]
            self.agentMemory[name] = in_view
            total_uncertainty = sum(self.uncertainty[t] for t in in_view) or 1.0
            obs[name]     = in_view.copy()
            rewards[name] = -total_uncertainty

            # --- flags for this agent ---
            terminations[name] = False
            truncations[name]  = (self.current_step >= self.max_steps)
            infos[name]        = {}

        # 2) Update all target uncertainties
        for t in self.target_names:
            covered = any(
                (pos == self.tPositions[t]).all()
                for pos in self.agentPositions.values()
            )
            delta = -1 if covered else +1
            self.uncertainty[t] = max(0, self.uncertainty[t] + delta)

        # 3) **Prune** any agent you flagged as done/truncated
        self.agents = [
            a for a in self.agents
            if not (terminations.get(a, False) or truncations.get(a, False))
        ]

    # 4) Set the global “episode over” flag
        terminations["__all__"] = False
        truncations["__all__"]  = len(self.agents) == 0

        return obs, rewards, terminations, truncations, infos

    def render(self):
        grid = np.full(self.grid_size, fill_value="·", dtype="<U1")
        for name, (x, y) in self.agentPositions.items():
            grid[y, x] = name.split('A')[-1]
        for name, (x,y) in self.tPositions.items():
            grid[y,x] = name.split('T')[-1]
        print("\n".join(" ".join(row) for row in grid))
        print()

    def close(self):
        pass
    def final_cost(self) -> float :
        # This is Jt, need to run at the end to determine Jt
        sum:float=0
        for target in self.target_names:
            sum+=self.uncertainty[target]
        if sum<=96 :
            print("Beat target")
        return sum

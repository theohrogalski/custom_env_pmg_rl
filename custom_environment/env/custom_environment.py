from pettingzoo.utils import parallel_to_aec
from pettingzoo.utils.env import ParallelEnv
import numpy as np
from gymnasium import spaces
from pettingzoo.test import parallel_api_test

class SimpleGridWorld(ParallelEnv):
    metadata = { "name": "pmg_env","render_modes": ["human"],}
    def __init__(self, num_agents=1, max_steps=100, num_targets=5, view_size=2, seed=None, grid_size=(10,10),render_mode=None):
        self.grid_size = grid_size 
        self.render_mode = render_mode or "human"
        self.max_steps = max_steps
        self.agent_names = [f"agent_{i}" for i in range(num_agents)]
        self.target_names = [f"target_{j}" for j in range(num_targets)]
        self.observation_spaces = {
            name: spaces.Box(0, 1, shape=(view_size,view_size), dtype=int)
            for name in self.agent_names
        }
        self.action_spaces = {
            name: spaces.Discrete(5)
            for name in self.agent_names
        }

        self.agentPositions:int = {}
        self.tPositions:int = {}
        self.uncertainty:int = {}
        self.current_step = None
        self.agents = None
        self.np_random = np.random.default_rng(seed)

    def reset(self, seed=None, options=None):
        self.current_step = 0
        # activate the first K agents
        if seed is not None :
            self.np_random = np.random.default_rng(seed)
        self.agents = self.agent_names.copy()
        # spawn positions
        for name in self.agents:
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
        for name in self.uncertainty:
            self.uncertainty[name] = 0
        obs = {name: self.agentPositions[name].copy().astype(int) for name in self.agents}
        infos = {name: {} for name in self.agents}
        return obs, infos

    def step(self, actions):
        self.current_step += 1
        obs, rewards, dones, infos, targets_in_space = {}, {}, {},{}, []
        radius = 2
        obs:dict = {}
        # actions has each agent and their corresponding action
        for name, act in actions.items():
            x, y = self.agentPositions[name]
            if act == 0:    # up
                y = min(y + 1, self.grid_size[1]-1)
            elif act == 1:  # down
                y = max(y - 1, 0)
            elif act == 2:  # left
                x = max(x - 1, 0)
            elif act == 3:  # right
                x = min(x + 1, self.grid_size[0]-1)
            elif act == 4: #stay still
                pass
            
            self.agentPositions[name] = np.array([x, y])
            obs_space_x, obs_space_y = {}, {}
            # first, the space around the agent is defined 
            low_x  = max(0, x-radius)
            high_x = min(self.grid_size[1], x+radius)
            low_y = max(0, y-radius)
            high_y = min(self.grid_size[1], y+radius)
            obs_space_x[name] = (low_x, high_x)
            obs_space_y[name] = (low_y,high_y)
            # Look at each target and if it is within range
            i:int=0
            for target in self.target_names: 
                i+=1
                xt, yt = self.tPositions[target]
                lowx, highx = obs_space_x[name]
                lowy, highy = obs_space_y[name]
                if xt>=lowx and xt<=highx and yt>=lowy and yt<=highy :
                    targets_in_space.append(target)
            target_dict = { agent: {} for agent in self.agents }
            if(len(targets_in_space)>0) :
                for b in range(len(targets_in_space)) :
                    target_dict[name][b] = targets_in_space[b]

            def meanUncertaintyValue(targets, name) -> float:
                uncertainty_sum = 0.0
                for t in targets:
                    uncertainty_sum += t.uncertainty
                print(f"{name} uncertainty:", uncertainty_sum)
                return uncertainty_sum

            # the observations are the uncertainty values, the rewards are the inverse
            obs[name] = meanUncertaintyValue(target_dict[name], name)
            rewards[name] = 1/meanUncertaintyValue(target_dict[name], name)
            dones[name] = self.current_step >= self.max_steps
            infos[name] = {}

        # if episode ended, mark all done
        if self.current_step >= self.max_steps:
            for name in dones:
                dones[name] = True

        return obs, rewards, dones, infos

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

# AEC wrapper
gridWorld = SimpleGridWorld(render_mode="human")

parallel_api_test(gridWorld)
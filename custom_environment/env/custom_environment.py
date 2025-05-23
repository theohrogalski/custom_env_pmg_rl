from pettingzoo.utils import parallel_to_aec
from pettingzoo.utils.env import ParallelEnv
import numpy as np
from gymnasium import spaces

class SimpleGridWorld(ParallelEnv):
    metadata = {"render_mode": ["human"], "name": "simple_grid"}

    def __init__(self, num_agents:int, max_steps:int, num_targets:int, view_size:int, grid_size=(None,None)):
        self.grid_size = grid_size
        self.num_agents = num_agents
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

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        # activate the first K agents
        self.agents = self.agent_names.copy()
        # spawn positions
        for name in self.agents:
            self.agentPositions[name] = self.np_random.integers(
                low=[0,0], high=self.grid_size, size=(2,)
            )
        self.targets = self.target_names.copy()
        for name in self.targets:
            #if you want a custom target setup plug in here
            self.tPositions[name] = self.np_random.integers(
                low=[0,0], high=self.grid_size, size = (2,)
            )
        # setting the uncertainty value
        for name in self.uncertainty:
            self.uncertainty[name] = 0
        obs = {name: self.positions[name].copy().astype(int) for name in self.agents}
        infos = {name: {} for name in self.agents}
        return obs, infos

    def step(self, actions):
        self.current_step += 1
        obs, obs_space_x, obs_space_y, rewards, dones, infos, targets_in_space = {}, {}, {}, {}, {},{}, []
        radius = 2
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
            # first, the space around the agent is defined   
            obs_space_x[name] = [max(0,x-radius), min(self.grid_size,x+radius)]
            obs_space_y[name] = [max(0,y-radius),min(self.grid_size,y+radius)]
            # Look at each target and if it is within rang
            for target in self.target_names: 
                xt, yt = self.tPositions[target]
                if xt>=obs_space_x[name][0] and xt<=obs_space_x[name][1] and yt>=obs_space_y[name][0] and yt<=obs_space_y[name][1] :
                    targets_in_space[name].append(target)

            def meanUncertaintyValue(self, targets_in_space) :
                uncertaintysum:int
                for target in targets_in_space :
                    uncertaintySum+=target.uncertainty
                # uncertaintySum
            rewards[name] = meanUncertaintyValue(targets_in_space[name])
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
SimpleGridWorldAEC = parallel_to_aec(SimpleGridWorld)


import functools
import random
from copy import copy

import numpy as np
from gymnasium.spaces import Discrete, MultiDiscrete

from pettingzoo import ParallelEnv


class CustomEnvironment(ParallelEnv):
    """The metadata holds environment constants.

    The "name" metadata allows the environment to be pretty printed.
    """

    metadata = {
        "name": "custom_environment_v0",
    }
    def __init__(self, numAgents:int, numTargets:int):
        """The init method takes in environment arguments.

        Should define the following attributes:
        - target x and y coordinates
        - agent 1,2,n... x and y coordinates
        - timestamp
        - possible_agents

        Note: as of v1.18.1, the action_spaces and observation_spaces attributes are deprecated.
        Spaces should be defined in the action_space() and observation_space() methods.
        If these methods are not overridden, spaces will be inferred from self.observation_spaces/action_spaces, raising a warning.

        These attributes should not be changed after initialization.
        """
        for i in range(numAgents) :
            #This is going to iterate through each agent and gives it no starting coordinates. Can be random
            setattr("self",f"agent{i}_x",None)
            setattr("self",f"agent{i}_y",None)
        for i in range(numTargets) :
            #Does the same thing but for each target, no starting values but assigned later

            setattr("self",f"target{i}_x",None)
            setattr("self",f"agent{i}_y",None)
            setattr("self", f"uncertainty_{i}",None)       
        self.timestep = None
        agentList:list=[]
        for j in range(numAgents):
            agentList.append(f"agent{j}")
        self.possible_agents = agentList

    def reset(self, numAgents:int, numTargets:int, seed=None, options=None):
        """Reset set the environment to a starting point.

        It needs to initialize the following attributes:
        - agents
        - timestamp
        - prisoner x and y coordinates
        - guard x and y coordinates
        - escape x and y coordinates
        - observation
        - infos

        And must set up the environment so that render(), step(), and observe() can be called without issues.
        """
        self.agents = copy(self.possible_agents)
        self.timestep = 0

        for k in range(numAgents):
            #Here is where the agents start
            agentStartingPointsX=[0]
            agentStartingPointsY=[0]
            #Here is where the targets start
            targetStartingPointsX=[1,2,3,4,5]
            targetStartingPointsY=[5,4,3,2,1]
            setattr("self", f"agent{k}_x",agentStartingPointsX[k])
            setattr("self", f"agent{k}_y",agentStartingPointsY[k])
        for g in range(numTargets) :
            setattr("self", f"target{k}_x",targetStartingPointsX[k])
            setattr("self", f"target{k}_y",targetStartingPointsY[k])


        observations = {
            a: (
                self.prisoner_x + 7 * self.prisoner_y,
                self.guard_x + 7 * self.guard_y,
                self.escape_x + 7 * self.escape_y,
            )
            for a in self.agents
        }

        # Get dummy infos. Necessary for proper parallel_to_aec conversion
        infos = {a: {} for a in self.agents}

        return observations, infos

    def step(self, actions, numAgents:int, numTargets:int):
        """Takes in an action for the current agent (specified by agent_selection).

        Needs to update:
        - prisoner x and y coordinates
        - guard x and y coordinates
        - terminations
        - truncations
        - rewards
        - timestamp
        - infos

        And any internal state used by observe() or render()
        """
        # Execute actions
        for m in range(numAgents) :
            f"agent{m}_action"=actions[f"agent{m}"]
        # 0 means stay, rest are left right up down
        for n in range(numAgents) :
            # Do nothing
            if f"agent{n}_action" == 0 :
                continue
            # Go left
            if f"agent{n}_action" == 1 :
                setattr("self",f"agent{n}_x",getattr("self",f"agent{n}_x")-1)
            # Go right
            if f"agent{n}_action" == 2 :
                setattr("self",f"agent{n}_x",getattr("self",f"agent{n}_x")+1)
            # Go up
            if f"agent{n}_action" == 3 :
                setattr("self",f"agent{n}_y",getattr("self",f"agent{n}_y")-1)
            # Go down
            if f"agent{n}_action" == 4 :
                setattr("self",f"agent{n}_y",getattr("self",f"agent{n}_y")+1)
        # Increase / Decrease uncertainty
        # For each target, check if each agents position is equal (blocked in masking so break is just for efficiency,, rewrite for speed)
        for n in range(numTargets) :
            for agentNum in range(agentNum) :
                if getattr("self",f"agent{agentNum}_x")==getattr("self",f"target{n}_x") and getattr("self",f"agent{agentNum}_y")==getattr("self",f"target{n}_y") :
                    setattr("self", f"uncertainty{numTargets}", getattr("self", f"uncertainty{numTargets}")-1)
                    break
            #in the no agent case
            setattr("self",f"uncertainty{n}", getattr("self",f"uncertainty{n}")+1)

       # The following block is commented out to avoid indentation errors.
       # if prisoner_action == 0 and self.prisoner_x > 0:
       #     self.prisoner_x -= 1
       # elif prisoner_action == 1 and self.prisoner_x < 6:
       #     self.prisoner_x += 1
       # elif prisoner_action == 2 and self.prisoner_y > 0:
       #     self.prisoner_y -= 1
       # elif prisoner_action == 3 and self.prisoner_y < 6:
       #     self.prisoner_y += 1
       #
       # if guard_action == 0 and self.guard_x > 0:
       #     self.guard_x -= 1
       # elif guard_action == 1 and self.guard_x < 6:
       #     self.guard_x += 1
       # elif guard_action == 2 and self.guard_y > 0:
       #     self.guard_y -= 1
       # elif guard_action == 3 and self.guard_y < 6:
       #     self.guard_y += 1

        # Check termination conditions
    
        terminations = {a: False for a in self.agents}
        rewards = {a: 0 for a in self.agents}
        # Reward structure where each agent is going to be rewarded on a non discrete way for uncertainty
        if self.prisoner_x == self.guard_x and self.prisoner_y == self.guard_y:
            rewards = {"prisoner": -1, "guard": 1}
            terminations = {a: True for a in self.agents}

        elif self.prisoner_x == self.escape_x and self.prisoner_y == self.escape_y:
            rewards = {"prisoner": 1, "guard": -1}
            terminations = {a: True for a in self.agents}

        # Check truncation conditions (overwrites termination conditions)
        truncations = {a: False for a in self.agents}
        if self.timestep > 100:
            rewards = {"prisoner": 0, "guard": 0}
            truncations = {"prisoner": True, "guard": True}
        self.timestep += 1

        # Get observations
        observations = {
            a: (
                self.prisoner_x + 7 * self.prisoner_y,
                self.guird_x + 7 * self.guard_y,
                self.escape_x + 7 * self.escape_y,
            )
            for a in self.agents
        }

        # Get dummy infos (not used in this example)
        infos = {a: {} for a in self.agents}

        if any(terminations.values()) or all(truncations.values()):
            self.agents = []

        return observations, rewards, terminations, truncations, infos

    def render(self):
        """Renders the environment."""
        grid = np.full((5, 5), " ")
        grid[self.prisoner_y, self.prisoner_x] = "P"
        grid[self.guard_y, self.guard_x] = "G"
        grid[self.escape_y, self.escape_x] = "E"
        print(f"{grid} \n")

    # Observation space should be defined here.
    # lru_cache allows observation and action spaces to be memoized, reducing clock cycles required to get each agent's space.
    # If your spaces change over time, remove this line (disable caching).
    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        # gymnasium spaces are defined and documented here: https://gymnasium.farama.org/api/spaces/
        return MultiDiscrete([7 * 7] * 3)

    # Action space should be defined here.
    # If your spaces change over time, remove this line (disable caching).
    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return Discrete(4)
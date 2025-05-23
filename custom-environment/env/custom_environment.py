import functools
import random
from copy import copy

import numpy as np
from gymnasium.spaces import Discrete, MultiDiscrete
from pettingzoo.utils.env import ParallelEnv

class gridEnv(ParallelEnv) :
    agents=
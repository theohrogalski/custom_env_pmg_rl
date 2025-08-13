from custom_environment_p import GridWithMemory
import agilerl
import gymnasium as gym
import numpy as np
import torch
from typing import List
from tqdm import trange
import matplotlib
from agilerl.algorithms import PPO
from agilerl.algorithms.core.registry import HyperparameterConfig, RLParameter
from agilerl.hpo.mutation import Mutations
from agilerl.training.train_on_policy import train_on_policy
from agilerl.hpo.tournament import TournamentSelection
from agilerl.utils.utils import create_population, make_vect_envs,observation_space_channels_to_first
from agilerl.rollouts.on_policy import collect_rollouts_recurrent
#env definition with defined key word arguments 
env = GridWithMemory(n_agents=4,max_steps=1000,num_targets=4,view_size=2,grid_size=(20,20))
env.reset(seed=1)
# Initial hyperparameters
INIT_HP = {
    "ALGO" : "PPO",
    "POP_SIZE": 4,  # Population size
    "BATCH_SIZE": 256,  # Batch size
    "LR": 0.001,  # Learning rate
    "LEARN_STEP": 1024,  # Learning frequency
    "GAMMA": 0.9,  # Discount factor
    "GAE_LAMBDA": 0.95,  # Lambda for general advantage estimation
    "ACTION_STD_INIT": 0.6,  # Initial action standard deviation
    "CLIP_COEF": 0.2,  # Surrogate clipping coefficient
    "ENT_COEF": 0.0,  # Entropy coefficient
    "VF_COEF": 0.5,  # Value function coefficient
    "MAX_GRAD_NORM": 0.5,  # Maximum norm for gradient clipping
    "USE_ROLLOUT_BUFFER": True, # Use a rollout buffer for data collection
    "TARGET_KL": None,  # Target KL divergence threshold
    "UPDATE_EPOCHS": 4,  # Number of policy update epochs
    "TARGET_SCORE": 200.0,  # Target score that will beat the environment
    "MAX_STEPS": 150000,  # Maximum number of steps an agent takes in an environment
    "EVO_STEPS": 10000,  # Evolution frequency
    "EVAL_STEPS": None,  # Number of evaluation steps per episode
    "EVAL_LOOP": 3,  # Number of evaluation episodes
    "TOURN_SIZE": 2,  # Tournament size
    "ELITISM": True,  # Elitism in tournament selection
}

# Mutation parameters
MUT_P = {
    # Mutation probabilities
    "NO_MUT": 0.4,  # No mutation
    "ARCH_MUT": 0.2,  # Architecture mutation
    "NEW_LAYER": 0.2,  # New layer mutation
    "PARAMS_MUT": 0.2,  # Network parameters mutation
    "ACT_MUT": 0.2,  # Activation layer mutation
    "RL_HP_MUT": 0.2,  # Learning HP mutation
    "MUT_SD": 0.1,  # Mutation strength
    "RAND_SEED": 42,  # Random seed
}
device = "cuda" if torch.cuda.is_available() else "cpu"

# Define the network configuration of a simple mlp with two hidden layers, each with 64 nodes


hp_config = HyperparameterConfig(
    lr = RLParameter(min=1e-4, max=1e-2),
    batch_size = RLParameter(
        min=8, max=1024, dtype=int
        )
)

# Number of environments
print(env.action_spaces)
from pprint import pprint

for agent in env.agents:
    print(f"Agent: {agent}")
    pprint(env.observation_spaces[agent])
# Configure the algo input arguments

action_spaces = [env.action_space(agent) for agent in env.agents]

action_space = action_spaces[0]

# Define a population
pop = create_population(
    algo=INIT_HP["ALGO"],  # RL algorithm
    observation_space=env.observation_spaces,  # State dimension
    action_space=action_space,  # Action dimension
    net_config={
    
},
    INIT_HP=INIT_HP,  # Initial hyperparameter
    hp_config=hp_config,  # RL hyperparameter configuration
    population_size=INIT_HP["POP_SIZE"],  # Population size
    num_envs=1,
    device=device,
)
# RL hyperparameters configuration for mutation during training

mutations = Mutations(
    no_mutation=MUT_P["NO_MUT"],
    architecture=MUT_P["ARCH_MUT"],
    new_layer_prob=MUT_P["NEW_LAYER"],
    parameters=MUT_P["PARAMS_MUT"],
    activation=MUT_P["ACT_MUT"],
    rl_hp=MUT_P["RL_HP_MUT"],
    mutation_sd=MUT_P["MUT_SD"],
    rand_seed=MUT_P["RAND_SEED"],
    device="cuda",
)
tournament = TournamentSelection(
    INIT_HP["TOURN_SIZE"],
    INIT_HP["ELITISM"],
    INIT_HP["POP_SIZE"],
    INIT_HP["EVAL_LOOP"],
)

# Save path for the best agent 
save_path = "elite_agent_ppo.pt"

trained_pop, pop_fitnesses = train_on_policy(
    env=env,
    env_name="PendulumPO-v1",
    algo="PPO",
    pop=pop,
    INIT_HP=INIT_HP,
    MUT_P=MUT_P,
    max_steps=INIT_HP["MAX_STEPS"],
    evo_steps=INIT_HP["EVO_STEPS"],
    eval_steps=INIT_HP["EVAL_STEPS"],
    eval_loop=INIT_HP["EVAL_LOOP"],
    tournament=tournament,
    mutation=mutations,
    wb=False,  # Boolean flag to record run with Weights & Biases
    save_elite=True,  # Boolean flag to save the elite agent in the population
    elite_path=save_path,
)

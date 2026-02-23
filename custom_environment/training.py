from custom_graph_env import GraphEnv
import torch
import os
from torch.nn.modules.container import ParameterList
from torch.distributions import Categorical
from model_two import observation_processing_network
import logging
import numpy as np
import os
from matplotlib import pyplot as plt
from torch.nn.functional import mse_loss
from gymnasium.wrappers import RecordEpisodeStatistics
def save_marl_checkpoint(episode, obs_nets, unc_nets, optimizers,epoch, path="./checkpoints/"):
    # Create directory if it doesn't exist
    if not os.path.exists(path):
        os.makedirs(path)

    # We pack everything into one dictionary
    checkpoint = {
        'episode': episode,
        # Save every agent's model weights
        'obs_state_dict': {agent: net.state_dict() for agent, net in obs_nets.items()},
        'unc_state_dict': {agent: net.state_dict() for agent, net in unc_nets.items()},
        'opt_state_dict': {agent: opt.state_dict() for agent, opt in optimizers.items()},
    }

    # Save to a temporary file first, then rename (prevents corruption if job dies mid-save)
    temp_path = f"{path}checkpoint_latest_training_session_one.tmp"
    final_path = f"{path}checkpoint_ep_{episode}_training_session_one_{epoch}.pt"
    
    torch.save(checkpoint, temp_path)
    os.rename(temp_path, final_path)
    
    # Also keep a 'latest' pointer for easy reloading
    torch.save(checkpoint, f"{path}latest.pt")
    print(f"--- Checkpoint saved at Episode {episode} ---")
logger = logging.getLogger("logger_train")
logging.basicConfig(filename='logger_train.log', level=logging.INFO)
print("logger created")
logger.info("------ Logger Started ------")
def calculate_unc_est_loss(predicted_value, actual_value):
    return (predicted_value-actual_value)^2

cur_length_list = []
def save_diagnostic_plots(step, agent_id,reward_history,epoch):
    """
    Saves a diagnostic figure to the /results folder.
    """
    
    # Use 'Agg' backend for headless cluster environments
    import matplotlib
    matplotlib.use('Agg') 
    
    fig = plt.plot(reward_history)
    
    plt.tight_layout()
    plt.savefig(f"./results/diag_agent_{agent_id}_step_{step}_{epoch}.png")
    plt.close()
def save_diagnostic_plots_total(step, agent_id,reward_history,epoch):
    """
    Saves a diagnostic figure to the /results folder.
    """
    
    # Use 'Agg' backend for headless cluster environments
    import matplotlib
    matplotlib.use('Agg') 
    
    fig = plt.plot(reward_history)
    
    plt.tight_layout()
    plt.savefig(f"./results/diag_agent_{agent_id}_step_{step}_{epoch}.png")
    plt.close()

def compute_ac_loss(log_prob, value, reward, next_value, done, gamma=0.99):
    # 1. Calculate Target (TD Target)
    # If done, next value is 0
    mask = 1 - int(done)
    target = reward + (gamma * next_value * mask)
    
    # 2. Calculate Advantage (Target - Baseline)
    # Detach target because we don't want to backprop through the target for the critic
    advantage = target - value
    
    # 3. Actor Loss: -log_prob * advantage
    actor_loss = -log_prob * advantage
    
    # 4. Critic Loss: MSE(value, target)
    # Use SmoothL1 or MSE
    print(torch.Tensor([target]).shape)
    print(torch.Tensor([value]).shape)
    critic_loss = mse_loss(torch.Tensor([value]), torch.Tensor([target]))
    
    # 5. Total Loss
    total_loss = actor_loss + (0.5 * critic_loss)
    
    return total_loss
num_nodes=50
env = GraphEnv(num_nodes=num_nodes)
#print(f" here3 {env.graph.nodes()}")
#print(env.agent_position
if torch.cuda.is_available():
    dev="cuda"
else:
    dev="cpu"
obs_nets:dict = {agent:observation_processing_network(env.graph.number_of_nodes()) for agent in env.possible_agents}
for nets in obs_nets.values():
    print("a")
    nets.to(dev)
optimizers = {agent:torch.optim.Adam(obs_nets[agent].parameters()) for agent in env.agents}

gamma = 0.99

critic_loss_dict:dict = {}
#print("starting")
import logging
reward_total = 0

# Hyperparameters
GAMMA = 0.99
critic_loss_dict = {}
# Main Episode Loop
reward_history:dict = {agent:[] for agent in env.possible_agents}

while env.agents:
    if env.num_moves==9999:
        reward_history:dict = {agent:[] for agent in env.possible_agents}
    if env.num_moves%2000 ==0 and env.num_moves !=0:
        save_marl_checkpoint(episode=env.num_moves,obs_nets=obs_nets,unc_nets=env.agent_to_net,optimizers=optimizers,epoch=env.num_epochs)
        for ag in env.agents:
            save_diagnostic_plots(step=env.num_moves,agent_id=ag,reward_history=reward_history[agent],epoch=env.num_epochs)
    actions={}
    step_data={}
    # --- PHASE 1: COLLECT ACTIONS ---
    for agent in env.agents:
    # 1. Run the forward pass

        logits, value, x_state, edges = obs_nets[agent](env.mental_map[agent], env.action_mask_to_node[int(agent[6:])],env.agent_to_net[agent])
        unc_net = env.agent_to_net[agent]
        # This call now only handles the GCN logic
        unc_loss = unc_net.update_estimator(x_state.detach(), edges)

        # 3. Handle RL sampling
        dist = Categorical(logits=logits)
        action_tensor = dist.sample()
        action_tensor=action_tensor.float()
        actions[agent] = action_tensor
        # Store for Phase 3
        step_data[agent] = {
            "log_prob": dist.log_prob(action_tensor),
            "value": value,
            "prediction": unc_net(x_state, edges).detach() # For Task 2 (DCBF)
        }
    
    
    # --- PHASE 2: STEP ENVIRONMENT ---
    
    obs, rewards, terminations, truncations, infos = env.step(actions)


        # Logging
        
    # Logging
    for agent in env.agents:
        reward_history[agent].append(rewards[agent])
 #   logging.info(f"Rewards: {list(rewards.values())}")
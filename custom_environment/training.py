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
def save_marl_checkpoint(episode, obs_nets, unc_nets, optimizers, path="./checkpoints/"):
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
    temp_path = f"{path}checkpoint_latest.tmp"
    final_path = f"{path}checkpoint_ep_{episode}.pt"
    
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
def save_diagnostic_plots(step, agent_id, true_unc, pred_unc, reward_history, loss_history):
    """
    Saves a diagnostic figure to the /results folder.
    """
    # Use 'Agg' backend for headless cluster environments
    import matplotlib
    matplotlib.use('Agg') 
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # --- Plot 1: Mental Map Accuracy (Node-by-Node) ---
    nodes = np.arange(len(true_unc))
    ax1.bar(nodes - 0.2, true_unc, width=0.4, label='Ground Truth', color='gray', alpha=0.5)
    ax1.bar(nodes + 0.2, pred_unc, width=0.4, label='GCN Prediction', color='blue', alpha=0.7)
    ax1.set_title(f"Agent {agent_id} | Mental Map Accuracy (Step {step})")
    ax1.set_xlabel("Node ID")
    ax1.set_ylabel("Uncertainty Value")
    ax1.set_ylim(0, 1.1)
    ax1.legend()

    # --- Plot 2: Training Progress ---
    ax2.plot(loss_history, label='Estimator Loss (MSE)', color='red')
    ax2.set_ylabel('Loss', color='red')
    ax2.tick_params(axis='y', labelcolor='red')
    
    ax2_twin = ax2.twinx()
    ax2_twin.plot(reward_history, label='Cumulative Reward', color='green')
    ax2_twin.set_ylabel('Reward', color='green')
    ax2_twin.tick_params(axis='y', labelcolor='green')
    
    ax2.set_title(f"Training Progress (Agent {agent_id})")
    ax2.set_xlabel("Steps (x100)")
    
    plt.tight_layout()
    plt.savefig(f"./results/diag_agent_{agent_id}_step_{step}.png")
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
    critic_loss = mse_loss(torch.Tensor([value]), torch.Tensor([target]))
    
    # 5. Total Loss
    total_loss = actor_loss + (0.5 * critic_loss)
    
    return total_loss
num_nodes=50
env = GraphEnv(num_nodes=num_nodes)
#print(f" here3 {env.graph.nodes()}")
#print(env.agent_position)

obs_nets:dict = {agent:observation_processing_network(env.graph.number_of_nodes()) for agent in env.possible_agents}
optimizers = {agent:torch.optim.Adam(obs_nets[agent].parameters()) for agent in env.agents}
gamma = 0.99

critic_loss_dict:dict = {}
#print("starting")
import logging
est_loss_history = []
reward_total = 0
# Hyperparameters
GAMMA = 0.99
critic_loss_dict = {}
# Main Episode Loop
while env.agents:
    if env.num_episodes %1000==0 and env.num_episdoes!=0:
        save_marl_checkpoint(env.episode_num,obs_nets,env.agent)
    actions={}
    step_data={}
    # --- PHASE 1: COLLECT ACTIONS ---
    for agent in env.agents:
    # 1. Run the forward pass
    # NOTE: We now get the [50, 5] state (x_state) back from the forward pass

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
    
    # TRIGGER FIGURE GENERATION
    # --- PHASE 2: STEP ENVIRONMENT ---
    # We step ONCE with all actions
    # Note: PettingZoo returns dicts keyed by agent
    
    obs, rewards, terminations, truncations, infos = env.step(actions)


        # Logging
        
    # Logging
    logging.info(f"Rewards: {list(rewards.values())}")
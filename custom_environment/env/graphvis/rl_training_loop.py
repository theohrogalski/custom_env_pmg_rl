
from custom_graph_env import GraphEnv
import supersuit as ss 
from stable_baselines3.common.vec_env import VecMonitor
from stable_baselines3.dqn import DQN
from stable_baselines3.common.evaluation import evaluate_policy


env=GraphEnv(num_nodes = 40, num_agents=6)
graph = env.create_custom_nx_graph(num_nodes=40)

vec_env = ss.pettingzoo_env_to_vec_env_v1(env)
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
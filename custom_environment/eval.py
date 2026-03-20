import torch
from custom_graph_env import GraphEnv
import random
from matplotlib import pyplot as plt
import torch
from model_two import observation_processing_network
#agents, optimizers = torch.load("")
import itertools#test_env = GraphEnv()
from torch.distributions import Categorical

from neural_model import uncertainty_estimator as ue 
class eval_type():
    def __init__(self,ckpt:str):
        self.ckpt= ckpt
        self.device= "cuda"
    def get_legal_actions(self,cur_node,mask)->list:
        count=0
        legal_moves:list=[]
        for value in mask[cur_node]:
            if value==1:
                legal_moves.append(count)

            count+=1
        return legal_moves
    def sit_on_nodes(self):
        env=GraphEnv()
        action_freeze={agent:0 for agent in env.possible_agents}
        
        actions={}
        uncertainty_history = []
        max_iters=10
        num_iters=0

        total_uncertainty_ever=0
        while env.agents and num_iters<max_iters:
            if env.num_moves%750==0 and env.num_moves!=0:
                env.reset()
                total_uncertainty_ever+=sum(uncertainty_history)
                uncertainty_history=[]
                num_iters+=1
                print(f"Num iters for sit {num_iters+1}/100")
            for agent in env.agents:
                if action_freeze[agent]==0:
                    if env.graph.nodes[env.agent_position[agent]]["target"]==1 and env.graph.nodes[env.agent_position[agent]]["agent_presence"]==0:
                        action_freeze[agent]=1
                        actions[agent] = torch.tensor(env.agent_position[agent])
                    else:
                        actions[agent]=torch.tensor(random.sample(self.get_legal_actions(env.agent_position[agent],env.action_mask_to_node),1))            

                else:
                    actions[agent] = torch.tensor(env.agent_position[agent])
            _, _, _,_, _ = env.step(actions)

            uncertainty_history.append(env.tot_unc)

        return (total_uncertainty_ever,env.longest_time_without_a_visit)


    def full_model(self):
        env = GraphEnv()
        "Evaluation for a pre-determined checkpoint for the full algorithm. Uses the self.ckpt as the path for loading models."
        check_dict  = torch.load(f"./checkpoints/{self.ckpt}.pt")
        
        obs_nets = check_dict["obs_state_dict"]
        unc_nets = check_dict["unc_state_dict"]
        opt = check_dict["opt_state_dict"]

        uncertainty_history = []
        max_iters=10
        num_iters=0
        total_uncertainty_ever = 0
        obs_net:dict = {agent:observation_processing_network(env.graph.number_of_nodes()) for agent in env.possible_agents}
        for agent,model in obs_net.items():
            model.load_state_dict(obs_nets[agent])
            model.to(self.device)
        agent_to_net:dict = {agent:ue(5,out_dim=1,hidden_dim=5) for agent in env.possible_agents}
        for agent, model in agent_to_net.items():
            
            model.load_state_dict(unc_nets[agent])
            model.to(self.device)
        actions={}
        while env.agents and num_iters<max_iters:
            if env.num_moves%750==0 and env.num_moves!=0:
                env.reset()
                total_uncertainty_ever+=sum(uncertainty_history)
                uncertainty_history=[]
                num_iters+=1
                print(f"Num iters for sit {num_iters}/100")
            for agent in env.agents:
                logits, value, x_state, edges = obs_net[agent](env.mental_map[agent], env.action_mask_to_node[int(agent[6:])],agent_to_net[agent], num_moves=env.num_moves)

                dist = Categorical(logits=logits)
                actions[agent] = dist.sample()

            _,_,_,_,_ = env.step(actions)
            uncertainty_history.append(env.tot_unc)
        print(f"Total uncertainty for {self.ckpt} is {total_uncertainty_ever}")
    def partial_model(self,data):
        statistics=[]
    def random(self):
        env=GraphEnv()
        actions={}
        uncertainty_history = []
        max_iters = 10
        num_iters=0
        total_uncertainty_ever=0
        while env.agents and num_iters<max_iters:
            if env.num_moves%200==0 and env.num_moves!=0:
                #plt.plot(uncertainty_history)
                total_uncertainty_ever+=sum(uncertainty_history)
                #plt.set_title("uncertainty_history_random")
                #plt.show()
                env.reset()
                uncertainty_history=[]
                num_iters+=1
                print(f"Num iters for Random at {num_iters}/100")
            for agent in env.agents:
                actions[agent]=torch.tensor(random.sample(self.get_legal_actions(env.agent_position[agent],env.action_mask_to_node),1))
            _, _, _,_, _ = env.step(actions)
            uncertainty_history.append(env.tot_unc)

        return total_uncertainty_ever
#eto : eval type object
def automatic_training_loop(eto):
    #print("Starting automatic evaluation loop...")
    #sit_num=eto.sit_on_nodes()
    #random_num = eto.random()
    #print(f"Random: {random_num}, Sit:{sit_num} ")
    eto.full_model()


if __name__=="__main__":
    eto = eval_type("checkpoint_ep_749_training_session_two_2")
    automatic_training_loop(eto)
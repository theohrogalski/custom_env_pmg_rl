import torch
from custom_graph_env import GraphEnv
import random
from matplotlib import pyplot as plt
import logging
import networkx as nx
from time import time
from custom_environment.models_full_model_d import observation_processing_network
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
        logger = logging.getLogger(f"log_sit")
        logging.basicConfig(filename=f'log_sit.log', level=logging.INFO)

        nodes_for_data=[100]
        num_agents_for_testing=[1,2,4,15]
        for nodes_num in nodes_for_data:
            for agents_num in num_agents_for_testing:
                time_start=time()
                env=GraphEnv(num_nodes=nodes_num,num_agents=agents_num)
                action_freeze={agent:0 for agent in env.possible_agents}
                
                actions={}
                uncertainty_history = []
                max_iters=20
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
                logger.info(f"{agents_num}_{nodes_num}")
                logger.info(total_uncertainty_ever)
                logger.info(env.longest_time_without_a_visit)
                logger.info(time()-time_start)



    def full_model(self,num_nodes,num_agents):
        logger = logging.getLogger(f"log_{num_nodes}_{num_agents}")    
        logging.basicConfig(filename=f'log_{num_nodes}_{num_agents}.log', level=logging.INFO)

        env = GraphEnv(num_nodes=num_nodes,num_agents=num_agents)
        "Evaluation for a pre-determined checkpoint for the full algorithm. Uses the self.ckpt as the path for loading models."
        check_dict  = torch.load(f"./checkpoints/checkpoint_ep_750_50_4_99_final_9.pt")
        
        obs_nets = check_dict["obs_state_dict"]
        unc_nets = check_dict["unc_state_dict"]
        opt = check_dict["opt_state_dict"]

        uncertainty_history = []
        max_iters=20
        num_iters=0
        total_uncertainty_ever = 0
        obs_net:dict = {agent:observation_processing_network(env.graph.number_of_nodes()) for agent in env.possible_agents}
        for agent,model in obs_net.items():
            model.load_state_dict(obs_nets[agent])
            model.to(self.device)
        agent_to_net:dict = {agent:ue(5,out_dim=1,hidden_dim=5,num_nodes=num_nodes) for agent in env.possible_agents}
        for agent, model in agent_to_net.items():
            
            model.load_state_dict(unc_nets[agent])
            model.to(self.device)
        actions={}
        while env.agents and num_iters<max_iters:
            time_start=time()
            if env.num_moves%750==0 and env.num_moves!=0:
                env.reset()
                total_uncertainty_ever+=sum(uncertainty_history)
                uncertainty_history=[]
                num_iters+=1
                print(f"num iters: {num_iters}")
                #print(f"Num iters for sit {num_iters}/20")
            for agent in env.agents:
                logits, value, x_state, edges = obs_net[agent](env.mental_map[agent], env.action_mask_to_node[int(agent[6:])],agent_to_net[agent], num_moves=env.num_moves,neighbors=env.action_mask_to_node[env.agent_position[agent]],position=env.agent_position[agent])
               # print(f"logits for agent {agent} are {logits}")
               # print(f"pos for agent {agent} are {env.agent_position[agent]}")

                dist = Categorical(logits=logits)
                actions[agent] = dist.sample()

            _,_,_,_,_ = env.step(actions)
            uncertainty_history.append(env.tot_unc)
        logger.info(f"{num_agents}_{num_nodes}")
        logger.info(total_uncertainty_ever)
        logger.info(env.longest_time_without_a_visit)
        logger.info(time()-time_start)
        return total_uncertainty_ever
    def partial_model(self,data):
        statistics=[]
    
    def random(self):
        logger = logging.getLogger(f"log_random")    
        logging.basicConfig(filename=f'log_random.log', level=logging.INFO)

        nodes_for_data=[20,50,100]
        num_agents_for_testing=[1,2,4,15]
        
        for nodes_num in nodes_for_data:
            for agents_num in num_agents_for_testing:
                time_start=time()
                env=GraphEnv(num_nodes=nodes_num,num_agents=agents_num)
                actions={}
                uncertainty_history = []
                max_iters = 20
                num_iters=0
                total_uncertainty_ever=0
                while env.agents and num_iters<max_iters:
                    if env.num_moves%750==0 and env.num_moves!=0:
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
                logger.info(f"{agents_num}_{nodes_num}")
                logger.info(total_uncertainty_ever)
                logger.info(env.longest_time_without_a_visit)
                logger.info(time()-time_start)
    def grazing(self):
        logger = logging.getLogger(f"log_grazing")    
        logging.basicConfig(filename=f'log_grazing.log', level=logging.INFO)

        nodes_for_data=[20,50,100]
        num_agents_for_testing=[1,2,4,15]
        
        for nodes_num in nodes_for_data:
            for agents_num in num_agents_for_testing:
                time_start=time()
                env=GraphEnv(num_nodes=nodes_num,num_agents=agents_num)
                agent_path_dict = {agent:[] for agent in env.possible_agents}
                actions={}
                uncertainty_history = []
                max_iters = 20
                num_iters=0
                total_uncertainty_ever=0
                while env.agents and num_iters<max_iters:
                    if env.num_moves%750==0 and env.num_moves!=0:
                        #plt.plot(uncertainty_history)
                        total_uncertainty_ever+=sum(uncertainty_history)
                        #plt.set_title("uncertainty_history_random")
                        #plt.show()
                        env.reset()
                        uncertainty_history=[]
                        num_iters+=1
                        print(f"Num iters for Random at {num_iters}/100")
                    for agent in env.agents:
                        if env.graph.nodes[env.agent_position[agent]]["uncertainty"]==0 and agent_path_dict[agent]==[]:
                            
                            max_unc=0
                            max_unc_node=env.agent_position[agent]
                            for node in env.graph.nodes():
                                for list in agent_path_dict.values():
                                    if list!=[]:
                                            if node==list[-1]:
                                                node=env.agent_position[agent]
                                if env.graph.nodes[node]["agent_presence"]==0:
                                    if env.graph.nodes[node]["uncertainty"]>max_unc:
                                        max_unc=env.graph.nodes[node]["uncertainty"]
                                        max_unc_node=node

                            agent_path_dict[agent]=nx.shortest_path(env.graph,env.agent_position[agent],target=max_unc_node)
                            actions[agent]=torch.tensor(agent_path_dict[agent].pop(0))
                          #  print(f"{agent} is going to {actions[agent]}")

            
                        elif agent_path_dict[agent] != []:
                            actions[agent]=torch.tensor(agent_path_dict[agent].pop(0))
                           # print(f"{agent} is going to {actions[agent]}")
                        else:
                            actions[agent]=torch.tensor(env.agent_position[agent])
                           # print(f"{agent} is staying")
                        



                    _, _, _,_, _ = env.step(actions)
                    uncertainty_history.append(env.tot_unc)
                logger.info(f"{agents_num}_{nodes_num}")
                logger.info(total_uncertainty_ever)
                logger.info(env.longest_time_without_a_visit)
                logger.info(time()-time_start)
#eto : eval type object
def automatic_training_loop(eto):
    #eto.random()

    #eto.sit_on_nodes()
    #eto.grazing()
    nodes_for_data=[50]
    num_agents_for_testing=[4]
    for nn in nodes_for_data:
        for ag in num_agents_for_testing:
            print(f"nn is {nn}")
            print(f"ag is {ag}")
            eto.full_model(num_agents=ag,num_nodes=nn)


if __name__=="__main__":
    eto = eval_type("checkpoint_ep_750_50_4_99_final_9")
    automatic_training_loop(eto)
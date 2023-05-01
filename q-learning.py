from collections import namedtuple, deque
from itertools import count
from environment import pharm_env
from environment import ReplayMemory
from se3nn import Se3NN
import numpy as np
import math
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch_geometric.data import Batch
import wandb
from copy import copy, deepcopy
import argparse

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def parse_arguments():
    parser = argparse.ArgumentParser()
    #q-learning parameters
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--gamma', type=float, default=0.7)
    parser.add_argument('--epsilon_start', type=float, default=0.9)
    parser.add_argument('--epsilon_decay', type=float, default=10000)
    parser.add_argument('--epsilon_min', type=float, default=0.01)
    parser.add_argument('--tau', type=float, default=1)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--memory_size', type=int, default=1000)
    parser.add_argument('--target_update', type=int, default=1)
    parser.add_argument('--num_episodes', type=int, default=10000)
    parser.add_argument('--num_steps', type=int, default=10)
    parser.add_argument('--num_tests', type=int, default=100)
    parser.add_argument('--num_test_steps', type=int, default=1000)
    #environment parameters
    parser.add_argument('--top_dir', type=str, default='../Pharmnn/data')
    parser.add_argument('--txt_file', type=str, default='../Pharmnn/data/pur2_dataset.txt')
    parser.add_argument('--randomize', type=bool, default=True)
    parser.add_argument('--reward', type=str, default='f1', choices=['f1','length'])
    parser.add_argument('--pharm_pharm_radius', type=float, default=10)
    parser.add_argument('--protein_pharm_radius', type=float, default=8)
    #model parameters
    parser.add_argument('--in_pharm_node_features',type=int,default=32)
    parser.add_argument('--in_prot_node_features',type=int,default=14)
    parser.add_argument('--sh_lmax',type=int,default=2)
    parser.add_argument('--ns',type=int,default=32)
    parser.add_argument('--nv',type=int,default=8)
    parser.add_argument('--num_conv_layers',type=int,default=4)
    parser.add_argument('--max_radius',type=float,default=8)
    parser.add_argument('--radius_embed_dim',type=int,default=50)
    parser.add_argument('--batch_norm',type=bool,default=True)
    parser.add_argument('--residual',type=bool,default=True)
    #run parameters
    parser.add_argument('--wandb_project', type=str, default='pharmrl')
    parser.add_argument('--wandb_run_name', type=str, default='')
    args = parser.parse_args()
    return args

def select_action(state,state_loader,policy_net,epsilon_start,epsilon_end,epsilon_decay,steps_done,test=False):
    
    if test:
        eps_threshold=0
    else:
        eps_threshold = epsilon_end + (epsilon_start - epsilon_end) * \
            math.exp(-1. * steps_done / epsilon_decay)
        steps_done += 1
        wandb.log({'epsilon':eps_threshold})

    values=[]
    if random.random() > eps_threshold:
        with torch.no_grad():
            for batch in state_loader:
                batch=batch.to(device)
                output=policy_net(batch)
                output=output.squeeze(-1)
                values=values+output.tolist()
            index=values.index(max(values))
            if index==len(state_loader.dataset)-1:
                #if terminated without building a graph
                if len(state_loader.dataset[index]['pharm'].index)<3:
                    values.pop(-1)
                    index=values.index(max(values))
                    return state_loader.dataset[index],steps_done
                else:
                    return state,steps_done
    else:
        #account for termination of graph
        index=random.randrange(len(state_loader.dataset))
        if index==len(state_loader.dataset)-1:
            #if terminated without building a graph
            if state is None or len(state['pharm'].index)<3:
                index=random.randrange(len(state_loader.dataset)-1)
                return state_loader.dataset[index],steps_done
            else:
                return state,steps_done
    return state_loader.dataset[index],steps_done

def main():
    args = parse_arguments()
    wandb.init(project=args.wandb_project, name=args.wandb_run_name)
    wandb.config.update(args)
    
    
    
    env = pharm_env(max_steps=args.num_steps-1,top_dir=args.top_dir,txt_file=args.txt_file,batch_size=args.batch_size,randomize=args.randomize,pharm_pharm_radius=args.pharm_pharm_radius,protein_pharm_radius=args.protein_pharm_radius)

    policy_net = Se3NN(in_pharm_node_features=args.in_pharm_node_features, in_prot_node_features=args.in_prot_node_features, sh_lmax=args.sh_lmax, ns=args.ns, nv=args.nv, num_conv_layers=args.num_conv_layers, max_radius=args.max_radius, radius_embed_dim=args.radius_embed_dim, batch_norm=args.batch_norm, residual=args.residual).to(device)
    target_net = Se3NN(in_pharm_node_features=args.in_pharm_node_features, in_prot_node_features=args.in_prot_node_features, sh_lmax=args.sh_lmax, ns=args.ns, nv=args.nv, num_conv_layers=args.num_conv_layers, max_radius=args.max_radius, radius_embed_dim=args.radius_embed_dim, batch_norm=args.batch_norm, residual=args.residual).to(device)
    target_net.load_state_dict(policy_net.state_dict())

    optimizer = optim.Adam(policy_net.parameters(), lr=args.lr)

    Transition = namedtuple('Transition',
                        ('next_graph', 'next_state_dataloader', 'reward'))
    memory = ReplayMemory(args.memory_size)

    batch_size = args.batch_size
    gamma = args.gamma
    epsilon_start = args.epsilon_start
    epsilon_decay = args.epsilon_decay
    epsilon_min = args.epsilon_min
    tau= args.tau
    target_update = args.target_update
    num_episodes = args.num_episodes
    num_steps = args.num_steps
    num_tests = args.num_tests
    num_test_steps = args.num_test_steps
    
    f1_scores=[]

    def optimize_model():
        if len(memory) < batch_size:
            return
        transitions = memory.sample(batch_size)
        batch = Transition(*zip(*transitions))

        next_graph_batch = batch.next_graph
        next_state_dataloader = batch.next_state_dataloader
        reward_batch = batch.reward
        
        next_graph_batch = Batch.from_data_list(list(next_graph_batch))
        reward_batch = torch.tensor(list(reward_batch))

        non_final_state_dataloader = [s for s in batch.next_state_dataloader]

        #graph_batch = graph_batch.to(device)
        next_graph_batch = next_graph_batch.to(device)
        reward_batch = reward_batch.to(device)

        state_action_values = policy_net(next_graph_batch)

        next_state_values = torch.zeros(batch_size, device=device)
        
        with torch.no_grad():
            for i,state_dataloader in enumerate(non_final_state_dataloader): 
                target_values=[]
                if state_dataloader is None:
                    continue
                for target_batch in state_dataloader:
                    target_batch=target_batch.to(device)
                    target_output=target_net(target_batch)
                    target_values+=target_output.squeeze(-1).tolist()
                next_state_values[i] = max(target_values)
        
        expected_state_action_values = (next_state_values.unsqueeze(1) * gamma) + reward_batch.unsqueeze(1)

        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values)
        wandb.log({"loss": loss})
        optimizer.zero_grad()
        loss.backward()
        #torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
        optimizer.step()

    steps_done = 0
    best_test_f1=0

    for i_episode in range(num_episodes):
        state_loader = env.reset()
        state=None
        prev_f1=0
        cum_reward=0
        for t in range(num_steps):
            next_state,steps_done = select_action(state,state_loader,policy_net,epsilon_start,epsilon_min,epsilon_decay,steps_done)
            next_state_loader, done, current_f1 = env.step(next_state,t)
            if args.reward_type=='f1':
                cum_reward=current_f1
            else:
                if prev_f1==current_f1:
                    reward=0
                elif current_f1>prev_f1:
                    reward=1
                else:
                    reward=0
                cum_reward+=reward
            prev_f1=current_f1
            #sanity check
            transition=Transition(next_state,next_state_loader,reward)
            memory.push(transition)
            state_loader = next_state_loader
            state=next_state
            optimize_model()
            if done:
                break
        if i_episode % target_update == 0:
            for param, target_param in zip(policy_net.parameters(), target_net.parameters()):
                target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)
        wandb.log({'graph_length':len(next_state['pharm'].index),'reward':cum_reward,'episode':i_episode,'f1_score':current_f1})
        #if i_episode % num_tests == 0:
        #    for env in test_envs



if __name__ == '__main__':
    main()

    

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
    parser.add_argument('--num_test_steps', type=int, default=10)
    parser.add_argument('--test_only', type=bool, default=False, help='test only')
    parser.add_argument('--stochastic', type=bool, default=False, help='stochastic policy')
    parser.add_argument('--beam_search', type=bool, default=False, help='beam search')
    parser.add_argument('--beam_size', type=int, default=3, help='beam size')
    parser.add_argument('--return_reward', type=str, default='dataframe', help='return reward')
    parser.add_argument('--min_features', type=int, default=3, help='minimum number of features in generated pharmacophore graph')
    #environment parameters
    parser.add_argument('--top_dir', type=str, default='../Pharmnn/data')
    parser.add_argument('--train_file', type=str, default='../Pharmnn/data/train_pharmrl_dataset.txt')
    parser.add_argument('--test_file', type=str, default='../Pharmnn/data/test_pharmrl_dataset.txt')
    parser.add_argument('--randomize', type=bool, default=True)
    parser.add_argument('--parallel', type=bool, default=True)
    parser.add_argument('--processes', type=int, default=4)
    parser.add_argument('--reward_type', type=str, default='f1', choices=['f1','length'])
    parser.add_argument('--pharm_pharm_radius', type=float, default=12)
    parser.add_argument('--protein_pharm_radius', type=float, default=8)
    parser.add_argument('--pharmit_database', type=str, default='./data/covid_moonshot/covid_moonshot')
    parser.add_argument('--actives_ism',type=str,default='./data/covid_moonshot/actives_smiles.ism')
    #model parameters
    parser.add_argument('--model_weights',type=str,default='')
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
    parser.add_argument('--save_model', type=bool, default=False)
    parser.add_argument('--save_path', type=str, default='')
    parser.add_argument('--out_file', type=str, default='')
    args = parser.parse_args()
    return args



def select_action(state,state_loader,policy_net,epsilon_start,epsilon_end,epsilon_decay,steps_done,test=False,stochastic=False,remove_state=False,time_step=0,min_features=3):
    
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
            #necessary for beam search
            if len(state_loader.dataset)==0:
                return None,steps_done,0
            for batch in state_loader:
                batch=batch.to(device)
                output=policy_net(batch)
                output=output.squeeze(-1)
                values=values+output.tolist()
            if stochastic:
                values=np.array(values)
                values[values<=0]=1e-10
                values=list(values)
                index=random.choices(range(len(values)),weights=values)[0]
            else:
                index=values.index(max(values))
            #beam search
            if remove_state:
                state_loader.dataset.pop(index)
                values.pop(index)
                if len(values)>0:
                    index=values.index(max(values))
                else:
                    return None,steps_done,0
            if index==len(state_loader.dataset)-1 and not remove_state:
                #if terminated without building a graph
                if len(state_loader.dataset[index]['pharm'].index)<min_features:
                    values.pop(-1)
                    index=values.index(max(values))
                    return state_loader.dataset[index],steps_done,values[index]
                else:
                    return state,steps_done,values[index]
            #graph length < 3 cant have value > 0 #for beam search
            if len(state_loader.dataset[index]['pharm'].index)<min_features and time_step>1:
                return state_loader.dataset[index],steps_done,0
    else:
        #account for termination of graph
        index=random.randrange(len(state_loader.dataset))
        if index==len(state_loader.dataset)-1:
            #if terminated without building a graph
            if state is None or len(state['pharm'].index)<min_features:
                index=random.randrange(len(state_loader.dataset)-1)
                return state_loader.dataset[index],steps_done,0
            else:
                return state,steps_done,0
    if len(values)>0:
        return state_loader.dataset[index],steps_done,values[index]
    else:
        return state_loader.dataset[index],steps_done,0


def remove_duplicates_nones(values,next_states,next_state_loaders,f1s,dones):
    new_values=[]
    new_next_states=[]
    new_next_state_loaders=[]
    new_f1s=[]
    new_dones=[]
    duplicate_value=values[0]
    #duplicate_value=next_states[0]
    new_values.append(values[0])
    new_next_states.append(next_states[0])
    new_next_state_loaders.append(next_state_loaders[0])
    new_f1s.append(f1s[0])
    new_dones.append(dones[0])
    for i in range(1,len(values)):
        if values[i]==duplicate_value:
            continue
        if next_states[i] is None:
            continue
        new_values.append(values[i])
        new_next_states.append(next_states[i])
        new_next_state_loaders.append(next_state_loaders[i])
        new_f1s.append(f1s[i])
        new_dones.append(dones[i])
        duplicate_value=values[i]
        #duplicate_state=next_states[i]
    return new_values,new_next_states,new_next_state_loaders,new_f1s,new_dones
        



def beam_search(env,gamma,state,state_loader,policy_net,beam_size,epsilon_start,epsilon_min,epsilon_decay,steps_done,num_test_steps,return_reward,args):
    
    values=[]
    next_states=[]
    f1s=[]
    dones=[]
    next_state_loaders=[]
    next_states.append(state)
    values.append(0)
    f1s.append(0)
    dones.append(False)
    
    next_state_loaders.append(state_loader)
    for t in range(num_test_steps):
        values_beam=[]
        next_states_beam=[]
        next_state_loaders_beam=[]
        f1s_beam=[]
        dones_beam=[]
        for i in range(len(next_states)):
            for j in range(beam_size):
                if not dones[i]:
                    remove_state=False
                    if j>0:
                        remove_state=True
                    next_state,steps_done,value=select_action(next_states[i],next_state_loaders[i],policy_net,epsilon_start,epsilon_min,epsilon_decay,steps_done,test=True,stochastic=False,remove_state=remove_state,time_step=t,min_features=args.min_features)
                    values_beam.append(value)
                    next_states_beam.append(next_state)
                    if next_state is None:
                        f1s_beam.append(0)
                        dones_beam.append(True)
                        next_state_loaders_beam.append(None)
                    else:
                        next_state_loader, done, current_f1 = env.step(next_state,t,test=True,current_graph=next_states[i],return_reward=return_reward,pharmit_database=args.pharmit_database,actives_ism=args.actives_ism)
                        f1s_beam.append(current_f1)
                        dones_beam.append(done)
                        next_state_loaders_beam.append(next_state_loader)
                else:
                    values_beam.append(values[i])
                    next_states_beam.append(next_states[i])
                    f1s_beam.append(f1s[i])
                    dones_beam.append(dones[i])
                    next_state_loaders_beam.append(next_state_loaders[i])
        values=values_beam
        next_states=next_states_beam
        next_state_loaders=next_state_loaders_beam
        f1s=f1s_beam
        dones=dones_beam
        values=np.array(values)
        values_index=np.argsort(-values)
        next_states=[next_states[i] for i in values_index]
        next_state_loaders=[next_state_loaders[i] for i in values_index]
        f1s=[f1s[i] for i in values_index]
        dones=[dones[i] for i in values_index]
        values=values[values_index]
        values=list(values)
        values,next_states,next_state_loaders,f1s,dones=remove_duplicates_nones(values,next_states,next_state_loaders,f1s,dones)
        if len(values)>beam_size:
            values=values[:beam_size]
            next_states=next_states[:beam_size]
            next_state_loaders=next_state_loaders[:beam_size]
            f1s=f1s[:beam_size]
            dones=dones[:beam_size]
        done_flag=True
        for done in dones:
            if not done:
                done_flag=False
                break
        if done_flag:
            break
    return next_states[0],steps_done,f1s[0],values[0]

def main():
    args = parse_arguments()
    wandb.init(project=args.wandb_project, name=args.wandb_run_name)
    wandb.config.update(args)
    
    env = pharm_env(max_steps=args.num_steps-1,top_dir=args.top_dir,train_file=args.train_file,test_file=args.test_file,batch_size=args.batch_size,randomize=args.randomize,pharm_pharm_radius=args.pharm_pharm_radius,protein_pharm_radius=args.protein_pharm_radius,parallel=args.parallel,pool_processes=args.processes)
    
    policy_net = Se3NN(in_pharm_node_features=args.in_pharm_node_features, in_prot_node_features=args.in_prot_node_features, sh_lmax=args.sh_lmax, ns=args.ns, nv=args.nv, num_conv_layers=args.num_conv_layers, max_radius=args.max_radius, radius_embed_dim=args.radius_embed_dim, batch_norm=args.batch_norm, residual=args.residual).to(device)
    target_net = Se3NN(in_pharm_node_features=args.in_pharm_node_features, in_prot_node_features=args.in_prot_node_features, sh_lmax=args.sh_lmax, ns=args.ns, nv=args.nv, num_conv_layers=args.num_conv_layers, max_radius=args.max_radius, radius_embed_dim=args.radius_embed_dim, batch_norm=args.batch_norm, residual=args.residual).to(device)
    if len(args.model_weights)>0:
        policy_net.load_state_dict(torch.load(args.model_weights))
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

    if args.test_only:
        num_episodes=1
    for i_episode in range(num_episodes):
        if not args.test_only:
            state_loader = env.reset(return_reward=args.return_reward)
            state=None
            prev_f1=0
            cum_reward=0
            for t in range(num_steps):
                next_state,steps_done,value = select_action(state,state_loader,policy_net,epsilon_start,epsilon_min,epsilon_decay,steps_done,min_features=args.min_features)
                next_state_loader, done, current_f1 = env.step(next_state,t,return_reward=args.return_reward,pharmit_database=args.pharmit_database,actives_ism=args.actives_ism)
                if args.reward_type=='f1':
                    reward=current_f1
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
                wandb.log({'f1_score':current_f1})
                #sanity check
                transition=Transition(next_state,next_state_loader,reward)
                memory.push(transition)
                state_loader = next_state_loader
                state=next_state
                if not args.test_only:
                    optimize_model()
                if done or len(state_loader.dataset)==0:
                    break
        if i_episode % target_update == 0 and not args.test_only:
            for param, target_param in zip(policy_net.parameters(), target_net.parameters()):
                target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)
        if not args.test_only:
            wandb.log({'graph_length':len(next_state['pharm'].index),'reward':cum_reward,'episode':i_episode})
        if i_episode % num_tests == 0 or args.test_only:
            with torch.no_grad():
                mean_test_f1=0
                out_file=None
                for i in range(len(env.systems_list[1])):
                    graphs_list=[]
                    values=[]
                    f1s=[]
                    state_loader = env.loop_test(return_reward=args.return_reward)
                    state=None
                    if args.beam_search:
                        state,steps_done,current_f1,value=beam_search(env,args.gamma,state,state_loader,policy_net,args.beam_size,epsilon_start,epsilon_min,epsilon_decay,steps_done,num_test_steps,args.return_reward,args)
                        values.append(value)
                        f1s.append(current_f1)
                        graphs_list.append(state['pharm'].pos.tolist())
                    else:
                        for t in range(num_test_steps):
                            if args.stochastic:
                                next_state,steps_done,value = select_action(state,state_loader,policy_net,epsilon_start,epsilon_min,epsilon_decay,steps_done,test=True,stochastic=True,min_features=args.min_features)
                            else:
                                next_state,steps_done,value = select_action(state,state_loader,policy_net,epsilon_start,epsilon_min,epsilon_decay,steps_done,test=True,min_features=args.min_features)
                            if args.test_only:
                                graphs_list.append(next_state['pharm'].pos.tolist())
                            next_state_loader, done, current_f1 = env.step(next_state,t,test=True,return_reward=args.return_reward,pharmit_database=args.pharmit_database,actives_ism=args.actives_ism)
                            state_loader = next_state_loader
                            state=next_state
                            if done or len(state_loader.dataset)==0:
                                if args.test_only:
                                    values.append(value)
                                    f1s.append(current_f1)
                                break
                    if args.test_only:
                        if len(args.out_file)>0:
                            if out_file is None:
                                out_file=open(args.out_file,'w')
                            out_file.write(str(env.system)+'\n')
                            out_file.write(str(values[-1])+'\n')
                            out_file.write(str(max(f1s))+'\n')
                            out_file.write(str(graphs_list)+'\n')
                        else:
                            print(env.system)
                            print(values[-1])
                            print(max(f1s))
                            print(graphs_list)
                    mean_test_f1+=current_f1
                mean_test_f1/=len(env.systems_list[1])
                wandb.log({'mean_test_f1':mean_test_f1})
                if mean_test_f1>best_test_f1:
                    best_test_f1=mean_test_f1
                    if args.save_model:
                        if len(args.save_path)==0:
                            torch.save(policy_net.state_dict(), wandb.run.dir+'/model.pt')
                        else:
                            torch.save(policy_net.state_dict(), args.save_path+'/model.pt')
                        print('saved model')
                    wandb.log({'best_test_f1':best_test_f1})
                    wandb.run.summary["best_test_f1"] = best_test_f1
                    print('best test f1: ',best_test_f1)
                    print('mean test f1: ',mean_test_f1)
                    print('episode: ',i_episode)



if __name__ == '__main__':
    main()

    

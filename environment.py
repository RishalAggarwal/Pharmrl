'''reinforcement learning environment for the pharm problem'''
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import molgrid
try:
    from molgrid.openbabel import pybel
except ImportError:
    from openbabel import pybel
import pandas as pd
from dataset import graphdataset
import pickle
from torch_geometric.data import DataLoader
from collections import namedtuple, deque
from itertools import count
import random

class MyCoordinateSet:
    
    def __init__(self, c):
        self.c = c
        
    def __getstate__(self):
        return self.c.coords.tonumpy(),self.c.type_index.tonumpy(), self.c.radii.tonumpy(), self.c.max_type,self.c.src
        
    def __setstate__(self,vals):    
        self.c = molgrid.CoordinateSet(vals[0],vals[1],vals[2],vals[3])

class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, transition):
        """Save a transition"""
        self.memory.append(transition)

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class pharm_env():
    
    def __init__(self,max_steps,top_dir,train_file,test_file,batch_size,randomize=True,pharm_pharm_radius=6,protein_pharm_radius=7):
        self.curent_step=0
        self.max_steps=max_steps
        self.dataset=None
        self.dataloader=None
        self.cache=None
        self.randomize=randomize
        self.batch_size=batch_size
        self.target_df=None
        self.pharm_pharm_radius=pharm_pharm_radius
        self.protein_pharm_radius=protein_pharm_radius
        s = molgrid.ExampleProviderSettings(data_root=top_dir)
        coord_reader = molgrid.CoordCache(molgrid.defaultGninaReceptorTyper,s)
        self.systems_list=[]
        self.coordcache = dict()
        self.system_to_cache={}
        for file in [train_file,test_file]:
            file_keys=[]
            data_info = pd.read_csv(file, header=None,delimiter=';') 
            self.top_dir = top_dir
            pdb_paths = np.asarray(data_info.iloc[:, -3])
            sdf_paths = np.asarray(data_info.iloc[:, -4])
            for pdbfile,sdffile in zip(pdb_paths,sdf_paths):
                if pdbfile not in self.coordcache:
                    self.coordcache[pdbfile] = MyCoordinateSet(coord_reader.make_coords(pdbfile))
                if sdffile not in self.coordcache:
                    self.coordcache[sdffile] = MyCoordinateSet(coord_reader.make_coords(sdffile))
                if pdbfile+'and'+sdffile in self.system_to_cache.keys():
                    continue
                else:
                    self.system_to_cache[pdbfile+'and'+sdffile]=[]
                    df_pdb=data_info[(data_info.iloc[:,-3]==pdbfile) & (data_info.iloc[:,-4]==sdffile)]
                    df_feats=df_pdb.iloc[:,-1]
                    df_feats=df_feats.apply(convert_to_list)
                    self.system_to_cache[pdbfile+'and'+sdffile].append({'label': np.asarray(df_pdb.iloc[:,0]),
                                        'centers':np.asarray(df_pdb.iloc[:,1:4]),
                                        'feature_vector': np.asarray(df_feats),
                                        'pdbfile': pdbfile,
                                        'sdffile': sdffile})   
                    file_keys.append(pdbfile+'and'+sdffile)
            self.systems_list.append(file_keys)
            #only randomize training systems
            if randomize and len(self.systems_list)==1:
                np.random.shuffle(self.systems_list[0])
        self.train_system_index=-1
        self.test_system_index=-1

    def reset(self):
        self.train_system_index+=1
        self.train_system_index=self.train_system_index%len(self.systems_list[0])
        self.system=self.systems_list[0][self.train_system_index]
        dir=self.system.split('anddude')[0].split('/')[-2]
        state_dataloader=self.create_state(self.systems_list[0][self.train_system_index],self.system_to_cache[self.systems_list[0][self.train_system_index]])
        
        self.target_df=pickle.load(open(self.top_dir+'/dude/all/'+dir+'/target_df.pkl','rb'))
        return state_dataloader
    
    def loop_test(self):

        self.test_system_index+=1
        self.test_system_index=self.test_system_index%len(self.systems_list[1])
        state_dataloader=self.create_state(self.systems_list[1][self.test_system_index],self.system_to_cache[self.systems_list[1][self.test_system_index]])
        self.system=self.systems_list[1][self.test_system_index]
        dir=self.system.split('anddude')[0].split('/')[-2]
        self.target_df=pickle.load(open(self.top_dir+'/dude/all/'+dir+'/target_df.pkl','rb'))
        return state_dataloader

    def create_state(self,system,cache,graph=None):
        '''return the state dataloader for the current system'''
        protein=self.coordcache[cache[0]['pdbfile']]
        protein_coords=protein.c.coords.tonumpy()
        protein_types=protein.c.type_index.tonumpy()
        pharm_coords=cache[0]['centers']
        pharm_feats=cache[0]['feature_vector']
        dataset=graphdataset(protein_coords,protein_types,pharm_coords,pharm_feats,current_graph=graph,pharm_pharm_radius=self.pharm_pharm_radius,protein_pharm_radius=self.protein_pharm_radius)
        self.current_graph=dataset.current_graph
        dataloader=DataLoader(dataset,batch_size=self.batch_size,shuffle=False)
        return dataloader

    def step(self,graph,current_step,test=False,current_graph=None,return_reward=True):
        '''return the reward and state dataloader and whether the episode is done for the next action'''
        if current_graph is not None:
            self.current_graph=current_graph
        if test:
            self.system_index=self.test_system_index
            systems_list_index=1
        else:
            self.system_index=self.train_system_index
            systems_list_index=0
        if current_step>=self.max_steps or graph==self.current_graph:
            done=True
        else:
            done=False
            reward=0
            next_state_dataloader=self.create_state(self.systems_list[systems_list_index][self.system_index],self.system_to_cache[self.systems_list[systems_list_index][self.system_index]],graph)
            self.current_graph=graph
            if len(next_state_dataloader.dataset)==0:
                done=True
        if return_reward:
            file_sets=[]
            system=self.systems_list[systems_list_index][self.system_index]
            cache=self.system_to_cache[system]
            pharm_coords=cache[0]['centers']
            num=len(graph['pharm'].index)
            target_df=self.target_df
            target_df['f1']=target_df['f1']/target_df['f1'].max()
            target_df=target_df[self.target_df['file'].str.startswith('pharmit_'+str(num)+'_')==1].sort_values(by='f1',ascending=False)
            for node in graph['pharm'].index:
                pos=pharm_coords[node]
                file_set=set(target_df[(np.abs(target_df['x']-pos[0])<5e-3)&(np.abs(target_df['y']-pos[1])<5e-3)&(np.abs(target_df['z']-pos[2])<5e-3)]['file'])
                file_sets.append(file_set)
            #TODO take the max reward
            interesection=set.intersection(*file_sets)
            max_f1=0
            for file in interesection:
                f1=self.target_df.loc[self.target_df['file']==file]['f1'].values[0]
                if f1>max_f1:
                    max_f1=f1
            reward=max_f1
        if done:
            next_state_dataloader=None
        return next_state_dataloader,done,reward
    
    def state_to_pharmit_query(graph,json_suffix,system):
        '''converts the state to a pharmit query'''        
        pharm_coords=graph['pharm'].index
        pharm_coords=pharm_coords.tolist()
        for coord in pharm_coords:
            for point in system:
                if np.allclose(coord,point['center']):
                    feature=point['label']
        '''pharm_coords=list(map(str,pharm_coords))
        pharm_coords=','.join(pharm_coords)
        with open(json_file_name,'w') as f:
            f.write('{"pharmacophores":[')
            f.write(pharm_coords)
            f.write(']}')
        return json_file_name'''

def convert_to_list(string):
    """converts a string of a list to a list"""
    string=string.replace('[','')
    string=string.replace(']','')
    string=string.replace(' ','')
    string=string.split(',')
    string=list(map(float,string))
    return string
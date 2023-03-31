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
    
    def __init__(self,max_steps,top_dir,txt_file,batch_size,randomize=True):
        self.curent_step=0
        self.max_steps=max_steps
        self.dataset=None
        self.dataloader=None
        self.cache=None
        self.randomize=randomize
        self.batch_size=batch_size
        self.target_df=None
        s = molgrid.ExampleProviderSettings(data_root=top_dir)
        coord_reader = molgrid.CoordCache(molgrid.defaultGninaReceptorTyper,s)
        data_info = pd.read_csv(txt_file, header=None,delimiter=';') 
        self.top_dir = top_dir
        pdb_paths = np.asarray(data_info.iloc[:, -3])
        sdf_paths = np.asarray(data_info.iloc[:, -4])
        self.coordcache = dict()
        self.system_to_cache={}
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
        self.systems = list(self.system_to_cache.keys())
        if randomize:
            np.random.shuffle(self.systems)
        self.system_index=-1

    def reset(self):
        self.system_index+=1
        self.system_index=self.system_index%len(self.systems)
        state_dataloader=self.create_state(self.systems[self.system_index],self.system_to_cache[self.systems[self.system_index]])
        system=self.systems[self.system_index]
        dir=system.split('and')[0].split('/')[-2]
        self.target_df=pickle.load(open(self.top_dir+'/dude/all/'+dir+'/target_df.pkl','rb'))
        return state_dataloader

    def create_state(self,system,cache,graph=None):
        '''return the state dataloader for the current system'''
        
        protein=self.coordcache[cache[0]['pdbfile']]
        protein_coords=protein.c.coords.tonumpy()
        protein_types=protein.c.type_index.tonumpy()
        pharm_coords=cache[0]['centers']
        pharm_feats=cache[0]['feature_vector']
        dataset=graphdataset(protein_coords,protein_types,pharm_coords,pharm_feats,current_graph=graph)
        self.current_graph=dataset.current_graph
        dataloader=DataLoader(dataset,batch_size=self.batch_size,shuffle=False)
        return dataloader

    def step(self,graph,current_step):
        '''return the reward and state dataloader and whether the episode is done for the next action'''
        if current_step>=self.max_steps or graph==self.current_graph:
            done=True
        else:
            done=False
            reward=0
            next_state_dataloader=self.create_state(self.systems[self.system_index],self.system_to_cache[self.systems[self.system_index]],graph)
            self.current_graph=graph
            if len(next_state_dataloader.dataset)==0:
                done=True
        if done:
            file_sets=[]
            system=self.systems[self.system_index]
            cache=self.system_to_cache[system]
            pharm_coords=cache[0]['centers']
            num=len(graph['pharm'].index)
            target_df=self.target_df
            print(target_df['f1'].max(),target_df.head(10))
            target_df['f1']=target_df['f1']/target_df['f1'].max()
            print(target_df.head(10))
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
            next_state_dataloader=None
        return next_state_dataloader,done,reward

def convert_to_list(string):
    """converts a string of a list to a list"""
    string=string.replace('[','')
    string=string.replace(']','')
    string=string.replace(' ','')
    string=string.split(',')
    string=list(map(float,string))
    return string
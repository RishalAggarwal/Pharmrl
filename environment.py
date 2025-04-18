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
from graphdataset import graphdataset
import pickle
from torch_geometric.data import DataLoader
from collections import namedtuple, deque
from itertools import count
import random
import json
import subprocess
import sys

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
    
    def __init__(self,max_steps,top_dir,train_file,test_file,batch_size,randomize=True,pharm_pharm_radius=6,protein_pharm_radius=7,parallel=False,pool_processes=1):
        
        self.parallel=parallel
        self.pool_processes=pool_processes
        self.curent_step=0
        self.max_steps=max_steps
        self.dataset=None
        self.dataloader=None
        self.cache=None
        self.randomize=randomize
        self.batch_size=batch_size
        self.target_df=None
        self.top_dir=top_dir
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
                    if pdbfile+'and'+sdffile in file_keys:
                        continue
                    file_keys.append(pdbfile+'and'+sdffile)
                    continue
                else:
                    self.system_to_cache[pdbfile+'and'+sdffile]=[]
                    df_pdb=data_info[(data_info.iloc[:,-3]==pdbfile) & (data_info.iloc[:,-4]==sdffile)]
                    grouped_df_pdb=df_pdb.groupby([1,2,3])
                    df_pdb=grouped_df_pdb.agg(lambda x: ':'.join(x)).reset_index()
                    df_feats=df_pdb.iloc[:,-1]
                    df_feats=df_feats.apply(lambda x: x.split(':')[0])
                    df_feats=df_feats.apply(convert_to_list)
                    self.system_to_cache[pdbfile+'and'+sdffile].append({'label': np.asarray(df_pdb.iloc[:,3]),
                                        'centers':np.asarray(df_pdb.iloc[:,0:3]),
                                        'feature_vector': np.asarray(df_feats),
                                        'pdbfile': pdbfile,
                                        'sdffile': sdffile})   
                    file_keys.append(pdbfile+'and'+sdffile)
            self.systems_list.append(file_keys)
            #only randomize training systems
            if randomize and len(self.systems_list)==1:
                np.random.shuffle(self.systems_list[0])
        self.systems_list[1]=sorted(self.systems_list[1])
        self.train_system_index=-1
        self.test_system_index=-1
        self.system_dir=None

    def get_target_df(self,return_reward='dataframe'):
        if return_reward=='dataframe':
            self.system_dir=self.system.split('and')[0].split('/')[-2]
            self.target_df=pickle.load(open(self.top_dir+'/'+self.system_dir+'/target_df.pkl','rb'))
        if return_reward=='dude_pharmit' or return_reward=='dude_ligand_pharmit':
            try:
                self.system_dir=self.system.split('crystal_ligand.mol2')[0].split('/')[0]
                self.target_df=pickle.load(open(self.top_dir+'/'+self.system_dir+'/target_df.pkl','rb'))
            except:
                self.system_dir=self.system.split('anddude')[0].split('/')[-2]
                self.target_df=pickle.load(open(self.top_dir+'/dude/all/'+self.system_dir+'/target_df.pkl','rb'))
        return self.target_df,self.system_dir

    def reset(self,return_reward='dataframe'):
        self.train_system_index+=1
        self.train_system_index=self.train_system_index%len(self.systems_list[0])
        self.system=self.systems_list[0][self.train_system_index]
        state_dataloader=self.create_state(self.systems_list[0][self.train_system_index],self.system_to_cache[self.systems_list[0][self.train_system_index]])
        self.target_df,self.system_dir=self.get_target_df(return_reward)
        return state_dataloader
    
    def loop_test(self,return_reward='dataframe'):

        self.test_system_index+=1
        self.test_system_index=self.test_system_index%len(self.systems_list[1])
        state_dataloader=self.create_state(self.systems_list[1][self.test_system_index],self.system_to_cache[self.systems_list[1][self.test_system_index]])
        self.system=self.systems_list[1][self.test_system_index]
        self.target_df=self.get_target_df(return_reward)
        return state_dataloader

    def create_state(self,system,cache,graph=None):
        '''return the state dataloader for the current system'''
        protein=self.coordcache[cache[0]['pdbfile']]
        protein_coords=protein.c.coords.tonumpy()
        protein_types=protein.c.type_index.tonumpy()
        pharm_coords=cache[0]['centers']
        pharm_feats=cache[0]['feature_vector']
        dataset=graphdataset(protein_coords,protein_types,pharm_coords,pharm_feats,current_graph=graph,pharm_pharm_radius=self.pharm_pharm_radius,protein_pharm_radius=self.protein_pharm_radius,parallel=self.parallel,pool_processes=self.pool_processes)
        self.current_graph=dataset.current_graph
        dataloader=DataLoader(dataset,batch_size=self.batch_size,shuffle=False,drop_last=False)
        return dataloader

    def step(self,graph,current_step,test=False,current_graph=None,return_reward='dataframe',pharmit_database='',actives_ism=''):
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
        system=self.systems_list[systems_list_index][self.system_index]
        num=len(graph['pharm'].index)
        cache=self.system_to_cache[system]
        if return_reward=='dataframe':
            file_sets=[]
            pharm_coords=cache[0]['centers']
            if type(self.target_df)==tuple:
                self.target_df=self.target_df[0]
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
        elif return_reward=='pharmit_query' or 'dude_pharmit' or 'dude_ligand_pharmit':
            if num<3:
                reward=0
            else:
                json_fname=self.state_to_pharmit_query(graph,'_pharmit',cache, ligand_only=(return_reward=='dude_ligand_pharmit'))
                pharmit_db_suffix=pharmit_database
                if return_reward=='dude_pharmit' or return_reward=='dude_ligand_pharmit':
                    if return_reward=='dude_ligand_pharmit':
                        pharmit_db_suffix='pharmit_ligand_db'
                    else:
                        pharmit_db_suffix='pharmit_db'
                    if 'dude' in self.top_dir:
                        pharmit_database=self.top_dir+'/'+self.system_dir+'/'+pharmit_db_suffix
                        actives_ism=self.top_dir+'/'+self.system_dir+'/actives_final.ism'
                    else:
                        pharmit_database=self.top_dir+'/dude/all/'+self.system_dir+'/'+pharmit_db_suffix
                        actives_ism=self.top_dir+'/dude/all/'+self.system_dir+'/actives_final.ism'
                decoys_ism=actives_ism.replace('actives','decoys')
                output=subprocess.check_output('python getf1.py '+json_fname+' '+pharmit_database+' --actives '+actives_ism+' --decoys '+decoys_ism,shell=True)
                output=output.decode()
                output_reward=output.split(' ')
                try:
                    reward=float(output_reward[1])
                except:
                    sys.exit('pharmit error')
                if return_reward=='dude_pharmit' or return_reward=='dude_ligand_pharmit':
                    if type(self.target_df)==tuple:
                        self.target_df=self.target_df[0]
                    reward=reward/self.target_df['f1'].max()
                    reward=min(reward,2 + (0.1*reward))
        if done:
            next_state_dataloader=None
        return next_state_dataloader,done,reward
    
    def state_to_pharmit_query(self,graph,json_suffix,system_cache,ligand_only=False):
        '''converts the state to a pharmit query'''        
        pharm_index=graph['pharm'].index
        pharm_index=pharm_index.tolist()
        cache=system_cache[0]
        pharmit_points={}
        pharmit_points["points"]=[]
        if not ligand_only:
            pharmit_points["exselect"] = "receptor"
            pharmit_points["extolerance"] = 1
            pharmit_points["recname"] = cache['pdbfile']
            pdb_file=open(self.top_dir+'/'+cache['pdbfile'],'r').readlines()
            pharmit_points["receptor"] = ''.join(pdb_file)
        for node in pharm_index:
            coord=cache['centers'][node]
            features=cache['label'][node]
            for feature in features.split(':'):
                radius=1
                if 'Hydrogen' in feature:
                    radius=1
                point_dict={"enabled": True,"name": feature, "radius":radius, "size": 1, "x":coord[0],"y":coord[1],"z":coord[2]}
                if ligand_only:
                    if type(self.target_df)==tuple:
                        target_df=self.target_df[0]
                    else:
                        target_df=self.target_df
                    pos=coord
                    vector=target_df[(np.abs(target_df['x']-pos[0])<5e-3)&(np.abs(target_df['y']-pos[1])<5e-3)&(np.abs(target_df['z']-pos[2])<5e-3)&(target_df['name']==feature)]['vector']
                    svector=target_df[(np.abs(target_df['x']-pos[0])<5e-3)&(np.abs(target_df['y']-pos[1])<5e-3)&(np.abs(target_df['z']-pos[2])<5e-3)&(target_df['name']==feature)]['svector'] 
                    if vector.values[0]==vector.values[0]:
                        point_dict['vector']=vector.values[0]
                    if svector.values[0]==svector.values[0]:
                        point_dict['svector']=svector.values[0]
                pharmit_points["points"].append(point_dict)
                if ligand_only:
                    break
        json_fname=self.top_dir+'/'+cache['sdffile'].split('.sdf')[0]+json_suffix+'.json'
        with open(json_fname,'w') as f:
            json.dump(pharmit_points,f)
        return json_fname

def convert_to_list(string):
    """converts a string of a list to a list"""
    string=string.replace('[','')
    string=string.replace(']','')
    string=string.replace(' ','')
    string=string.split(',')
    string=list(map(float,string))
    return string

class Inference_environment():

    def __init__(self,receptor,receptor_string,receptor_file_name,ligand_string,ligand_file_name,feature_points,cnn_hidden_features,batch_size,top_dir,pharm_pharm_radius=6,protein_pharm_radius=7,max_steps=10,parallel=False,pool_processes=1):
        self.receptor=receptor
        self.receptor_string=receptor_string
        self.receptor_file_name=receptor_file_name
        self.ligand_string=ligand_string
        self.ligand_format=ligand_file_name
        self.feature_points=feature_points
        self.starter_points=feature_points[:,4]
        self.cnn_hidden_features=cnn_hidden_features.detach().cpu().numpy()
        self.current_step=0
        self.max_steps=max_steps
        self.batch_size=batch_size
        self.top_dir=top_dir
        self.pharm_pharm_radius=pharm_pharm_radius
        self.protein_pharm_radius=protein_pharm_radius
        self.parallel=parallel
        self.pool_processes=pool_processes  
        self.current_graph=None

    def create_state(self,graph=None,starter_points=None):
        '''return the state dataloader for the current system'''
        protein=self.receptor
        protein_coords=protein.coords.tonumpy()
        protein_types=protein.type_index.tonumpy()
        pharm_coords=np.array(self.feature_points[:,1:4],dtype=np.float32)
        pharm_feats=self.cnn_hidden_features
        dataset=graphdataset(protein_coords,protein_types,pharm_coords,pharm_feats,current_graph=graph,pharm_pharm_radius=self.pharm_pharm_radius,protein_pharm_radius=self.protein_pharm_radius,parallel=self.parallel,pool_processes=self.pool_processes,starter_points=starter_points)
        self.current_graph=dataset.current_graph
        dataloader=DataLoader(dataset,batch_size=self.batch_size,shuffle=False,drop_last=False)
        return dataloader 

    def reset(self):
        self.current_step=0
        if self.starter_points.any():
            return self.create_state(starter_points=self.starter_points)
        return self.create_state()
    
    def step(self,graph,current_step):
        if current_step>=self.max_steps or graph==self.current_graph:
            done=True
        else:
            done=False
            next_state_dataloader=self.create_state(graph)
            self.current_graph=graph
            if len(next_state_dataloader.dataset)==0:
                done=True
        if done:
            next_state_dataloader=None
        return next_state_dataloader,done
    
    def state_to_json(self,graph,label=None,min_features=3):
        pharm_index=graph['pharm'].index
        if len(pharm_index)<min_features:
            print('not enough features around selected set to form pharmacophore of full size')
        pharm_index=pharm_index.tolist()
        pharmit_points={}
        pharmit_points["points"]=[]
        #if label is None or (not label=='model_ligand'): 
        pharmit_points["exselect"] = "receptor"
        pharmit_points["extolerance"] = 1
        pharmit_points["recname"] = 'receptor.pdb'
        pharmit_points["receptor"] = self.receptor_string
        pharmit_points["recname"]=self.receptor_file_name
        if self.ligand_string is not None:
            pharmit_points["ligand"]=self.ligand_string
            pharmit_points["ligandFormat"]=self.ligand_format
        for node in pharm_index:
            coord=self.feature_points[node,1:4]
            features=self.feature_points[node,0]
            vector=self.feature_points[node,-2]
            svector=self.feature_points[node,-1]
            for feature in features.split(':'):
                radius=1
                '''if 'Hydrogen' in feature:
                    radius=0.5
                if 'Aromatic' in feature:
                    radius=1.1
                if 'ion' in feature:
                    radius=0.75'''
                if vector is not None and (isinstance(vector, list) or isinstance(vector, dict)):
                    if label is not None:
                        point_dict={"enabled": True,"name": feature, "radius":radius,"x":coord[0],"y":coord[1],"z":coord[2],"vector":vector,"svector":svector,"label": label}
                    else:
                        point_dict={"enabled": True,"name": feature, "radius":radius,"x":coord[0],"y":coord[1],"z":coord[2],"vector":vector,"svector":svector}      
                else:
                    if label is not None:
                        point_dict={"enabled": True,"name": feature, "radius":radius,"x":coord[0],"y":coord[1],"z":coord[2],"label": label}
                    else:
                        point_dict={"enabled": True,"name": feature, "radius":radius,"x":coord[0],"y":coord[1],"z":coord[2]}
                pharmit_points["points"].append(point_dict)
        return pharmit_points


        


        
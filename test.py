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
import torch
from torch_geometric.data import HeteroData, Dataset
from scipy.spatial.distance import cdist
from copy import copy, deepcopy
import numpy as np
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
import wandb
import argparse



#dataset test
def test_dataset():
    protein_coords=np.array([[0.5,0.5,0.5],[2,2,2]])
    pharm_coords=np.array([[1,1,1],[3,3,3]])
    protein_types=np.array([[1],[13]])
    pharm_types=np.random.random((2,32))
    dataset=graphdataset(protein_coords,protein_types,pharm_coords,pharm_types,None)
    test_dataloader=DataLoader(dataset,batch_size=1,shuffle=False)
    for i,batch in enumerate(test_dataloader):
        assert batch[0]['pharm'].index ==torch.tensor([i],dtype=torch.long)
        assert (batch[0]['protein'].index ==torch.tensor([0,1],dtype=torch.long)).all()
        assert batch[0]['pharm'].x.shape ==torch.Size([1,32])
        assert batch[0]['protein'].x.shape ==torch.Size([2,1])
        assert (batch[0]['pharm'].pos ==torch.tensor(np.expand_dims(pharm_coords[i],axis=0),dtype=torch.float)).all()
        assert (batch[0]['protein'].pos ==torch.tensor(protein_coords,dtype=torch.float)).all()
        assert (batch[0]['protein', 'proteinpharm', 'pharm'].edge_index == torch.tensor([[0,1],[0,0]],dtype=torch.long)).all()
        if i==0:
            assert (batch[0]['protein', 'proteinpharm', 'pharm'].edge_attr == torch.tensor([[-0.5,-0.5,-0.5],[1,1,1]],dtype=torch.float)).all()
        elif i==1:
            assert (batch[0]['protein', 'proteinpharm', 'pharm'].edge_attr == torch.tensor([[-2.5,-2.5,-2.5],[-1,-1,-1]],dtype=torch.float)).all()
    
    protein_coords=np.array([[0.5,0.5,0.5],[2,2,2],[4,4,4],[100,100,100]])
    protein_types=np.array([[1],[13],[5],[4]])
    current_graph=test_dataloader.dataset[0]
    dataset=graphdataset(protein_coords,protein_types,pharm_coords,pharm_types,current_graph)
    test_dataloader=DataLoader(dataset,batch_size=1,shuffle=False)
    for i,batch in enumerate(test_dataloader):
        assert (batch[0]['pharm'].index ==torch.tensor([0,1],dtype=torch.long)).all()
        assert (batch[0]['protein'].index ==torch.tensor([0,1,2],dtype=torch.long)).all()
        assert batch[0]['pharm'].x.shape ==torch.Size([2,32])
        assert batch[0]['protein'].x.shape ==torch.Size([3,1])
        assert (batch[0]['pharm'].pos ==torch.tensor(pharm_coords,dtype=torch.float)).all()
        assert (batch[0]['protein'].pos ==torch.tensor(protein_coords[:3],dtype=torch.float)).all()
        assert (batch[0]['protein', 'proteinpharm', 'pharm'].edge_index == torch.tensor([[0,1,0,1,2],[0,0,1,1,1]],dtype=torch.long)).all()
        assert (batch[0]['protein', 'proteinpharm', 'pharm'].edge_attr == torch.tensor([[-0.5,-0.5,-0.5],[1,1,1],[-2.5,-2.5,-2.5],[-1,-1,-1],[1,1,1]],dtype=torch.float)).all()
        assert (batch[0]['pharm','pharmpharm','pharm'].edge_index == torch.tensor([[0,1],[1,0]],dtype=torch.long)).all()
        assert (batch[0]['pharm','pharmpharm','pharm'].edge_attr == torch.tensor([[-2,-2,-2],[2,2,2]],dtype=torch.float)).all()
        break
    current_graph=test_dataloader.dataset[0]
    dataset=graphdataset(protein_coords,protein_types,pharm_coords,pharm_types,current_graph)
    test_dataloader=DataLoader(dataset,batch_size=1,shuffle=False)
    assert len(test_dataloader)==1

    pharm_coords=np.array([[1,1,1],[3,3,3],[300,300,300]])
    pharm_types=np.random.random((3,32))
    dataset=graphdataset(protein_coords,protein_types,pharm_coords,pharm_types,current_graph)
    test_dataloader=DataLoader(dataset,batch_size=1,shuffle=False)
    assert len(test_dataloader)==1
    print('dataset test passed')

def test_model():
    protein_coords=np.array([[0.5,0.5,0.5],[2,2,2]])
    pharm_coords=np.array([[1,1,1],[3,3,3]])
    protein_types=np.array([[1],[13]],dtype=np.int64)
    pharm_types=np.random.random((2,32))
    dataset=graphdataset(protein_coords,protein_types,pharm_coords,pharm_types,None)
    test_dataloader=DataLoader(dataset,batch_size=2,shuffle=False)
    model=Se3NN()
    for i,batch in enumerate(test_dataloader):
        output=model(batch)
        assert output.shape ==torch.Size([2,1])
    current_graph=test_dataloader.dataset[0]
    dataset=graphdataset(protein_coords,protein_types,pharm_coords,pharm_types,current_graph)
    test_dataloader=DataLoader(dataset,batch_size=1,shuffle=False)
    for i,batch in enumerate(test_dataloader):
        output=model(batch)
        assert output.shape ==torch.Size([1,1])
    print('model test passed')


if __name__ == '__main__':
    test_dataset()
    test_model()
import torch
from torch_geometric.data import HeteroData, Dataset
from scipy.spatial.distance import cdist
from copy import copy, deepcopy
import numpy as np

class graphdataset(Dataset):
    
    def __init__(self,protein_coords,protein_types,pharm_coords,pharm_features,current_graph,pharm_pharm_radius=6,protein_pharm_radius=7):
        super(graphdataset).__init__()

        self.protein_coords = protein_coords
        self.protein_types = protein_types
        self.pharm_coords = pharm_coords
        self.pharm_features = pharm_features
        self.current_graph = current_graph
        self.protein_pharm_radius = protein_pharm_radius
        self.pharm_pharm_radius = pharm_pharm_radius
        self.datapoints = []
        
        if self.current_graph is None:
            self.current_graph=HeteroData()
            pharm_extension_nodes=range(pharm_coords.shape[0])
            self.datapoints  = self.generate_graphs(self.current_graph,pharm_extension_nodes)          
        else:
            current_graph_nodes = current_graph['pharm'].index
            current_pharm_coords=self.pharm_coords[current_graph_nodes]
            if len(current_pharm_coords.shape)==1:
                current_pharm_coords=np.expand_dims(current_pharm_coords,axis=0)
            pharm_pharm_distances = cdist(self.pharm_coords,current_pharm_coords)
            pharm_pharm_distances_min = pharm_pharm_distances.min(axis=1)
            pharm_extension_nodes = np.where(pharm_pharm_distances_min<pharm_pharm_radius)[0]
            self.datapoints = self.generate_graphs(self.current_graph,pharm_extension_nodes,pharm_pharm_distances)
            self.datapoints.append(self.current_graph)
        

    def generate_graphs(self,current_graph,pharm_extension_nodes,pharm_pharm_distances=None):
        #TODO ensure directionality of edges is correct, reverse edges
        datapoints=[]
        protein_pharm_distances = cdist(self.protein_coords,self.pharm_coords)
        if pharm_pharm_distances is None:
            for i in pharm_extension_nodes:
                current_graph=deepcopy(self.current_graph)
                current_graph['pharm'].index=torch.tensor([i],dtype=torch.long)
                current_graph['pharm'].x=torch.tensor(np.expand_dims(self.pharm_features[i],axis=0),dtype=torch.float32)
                current_graph['pharm'].pos=torch.tensor(np.expand_dims(self.pharm_coords[i],axis=0),dtype=torch.float32)
                protein_nodes=np.where(protein_pharm_distances[:,i]<self.protein_pharm_radius)[0]
                if len(protein_nodes)==0:
                    continue
                current_graph['protein'].index=torch.tensor(protein_nodes,dtype=torch.long)
                current_graph['protein'].x=torch.tensor(self.protein_types[protein_nodes],dtype=torch.long)
                current_graph['protein'].pos=torch.tensor(self.protein_coords[protein_nodes],dtype=torch.float32)
                protein_nodes_graph=np.arange(len(protein_nodes))
                current_graph['protein','proteinpharm','pharm'].edge_index=torch.tensor([protein_nodes_graph,[0]*len(protein_nodes)],dtype=torch.long)
                current_graph['protein','proteinpharm','pharm'].edge_attr=torch.tensor(self.protein_coords[protein_nodes]-self.pharm_coords[i],dtype=torch.float32)
                datapoints.append(current_graph)

        else:
            for i in pharm_extension_nodes:
                if i in current_graph['pharm'].index:
                    continue
                current_graph=deepcopy(self.current_graph)
                current_graph['pharm'].index=torch.cat((current_graph['pharm'].index,torch.tensor([i],dtype=torch.long)))
                current_graph['pharm'].x=torch.cat((current_graph['pharm'].x,torch.tensor(np.expand_dims(self.pharm_features[i],axis=0),dtype=torch.float32)))
                current_graph['pharm'].pos=torch.cat((current_graph['pharm'].pos,torch.tensor(np.expand_dims(self.pharm_coords[i],axis=0),dtype=torch.float32)))
                pharm_nodes=np.where(pharm_pharm_distances[i,:]<self.pharm_pharm_radius)[0]
                #pharm_nodes=current_graph['pharm'].index[pharm_nodes].tolist()
                try:
                    current_graph['pharm','pharmpharm','pharm'].edge_index=torch.cat((current_graph['pharm','pharmpharm','pharm'].edge_index,torch.tensor([pharm_nodes,[len(current_graph['pharm'].index)-1]*len(pharm_nodes)],dtype=torch.long)),dim=1)
                    current_graph['pharm','pharmpharm','pharm'].edge_index=torch.cat((current_graph['pharm','pharmpharm','pharm'].edge_index,torch.tensor([[len(current_graph['pharm'].index)-1]*len(pharm_nodes),pharm_nodes],dtype=torch.long)),dim=1)    
                    current_graph['pharm','pharmpharm','pharm'].edge_attr=torch.cat((current_graph['pharm','pharmpharm','pharm'].edge_attr,torch.tensor(self.pharm_coords[pharm_nodes]-self.pharm_coords[i],dtype=torch.float32)))
                    current_graph['pharm','pharmpharm','pharm'].edge_attr=torch.cat((current_graph['pharm','pharmpharm','pharm'].edge_attr,torch.tensor(self.pharm_coords[i]-self.pharm_coords[pharm_nodes],dtype=torch.float32)))
                except:
                    current_graph['pharm','pharmpharm','pharm'].edge_index=torch.tensor([pharm_nodes,[len(current_graph['pharm'].index)-1]*len(pharm_nodes)],dtype=torch.long)
                    current_graph['pharm','pharmpharm','pharm'].edge_index=torch.cat((current_graph['pharm','pharmpharm','pharm'].edge_index,torch.tensor([[len(current_graph['pharm'].index)-1]*len(pharm_nodes),pharm_nodes],dtype=torch.long)),dim=1)   
                    current_graph['pharm','pharmpharm','pharm'].edge_attr=torch.tensor(self.pharm_coords[pharm_nodes]-self.pharm_coords[i],dtype=torch.float32)
                    current_graph['pharm','pharmpharm','pharm'].edge_attr=torch.cat((current_graph['pharm','pharmpharm','pharm'].edge_attr,torch.tensor(self.pharm_coords[i]-self.pharm_coords[pharm_nodes],dtype=torch.float32)))
                protein_nodes=np.where(protein_pharm_distances[:,i]<self.protein_pharm_radius)[0]
                set1=set(protein_nodes)
                set2=set(current_graph['protein'].index.tolist())
                protein_nodes_intersection=list(set1&set2)
                if len(protein_nodes_intersection)>0:
                    #protein_nodes_graph=current_graph['protein'].index.tolist().index(protein_nodes_intersection)
                    protein_nodes_graph=[current_graph['protein'].index.tolist().index(x) for x in protein_nodes_intersection]
                    current_graph['protein','proteinpharm','pharm'].edge_index=torch.cat((current_graph['protein','proteinpharm','pharm'].edge_index,torch.tensor([protein_nodes_graph,[len(current_graph['pharm'].index)-1]*len(protein_nodes_graph)],dtype=torch.long)),dim=1)
                    current_graph['protein','proteinpharm','pharm'].edge_attr=torch.cat((current_graph['protein','proteinpharm','pharm'].edge_attr,torch.tensor(self.protein_coords[protein_nodes_intersection]-self.pharm_coords[i],dtype=torch.float32)))
                protein_nodes_new=list(set1-set2)
                if len(protein_nodes_new)==0:
                    datapoints.append(current_graph)
                    continue
                protein_nodes_graph=np.arange(len(current_graph['protein'].index.tolist()),len(current_graph['protein'].index.tolist())+len(protein_nodes_new))
                current_graph['protein','proteinpharm','pharm'].edge_index=torch.cat((current_graph['protein','proteinpharm','pharm'].edge_index,torch.tensor([protein_nodes_graph,[len(current_graph['pharm'].index)-1]*len(protein_nodes_graph)],dtype=torch.long)),dim=1)
                current_graph['protein','proteinpharm','pharm'].edge_attr=torch.cat((current_graph['protein','proteinpharm','pharm'].edge_attr,torch.tensor(self.protein_coords[protein_nodes_new]-self.pharm_coords[i],dtype=torch.float32)))
                current_graph['protein'].index=torch.cat((current_graph['protein'].index,torch.tensor(protein_nodes_new,dtype=torch.long)))
                current_graph['protein'].x=torch.cat((current_graph['protein'].x,torch.tensor(self.protein_types[protein_nodes_new],dtype=torch.long)))
                current_graph['protein'].pos=torch.cat((current_graph['protein'].pos,torch.tensor(self.protein_coords[protein_nodes_new],dtype=torch.float32)))
                datapoints.append(current_graph)
        return datapoints
    
    def __len__(self):
        return len(self.datapoints)
    
    def __getitem__(self,idx):
        return self.datapoints[idx]
    
    def get(self,idx):
        return self.datapoints[idx]

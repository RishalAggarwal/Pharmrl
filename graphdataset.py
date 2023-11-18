import torch
from torch_geometric.data import HeteroData, Dataset
from scipy.spatial.distance import cdist
from copy import copy, deepcopy
import numpy as np
from multiprocess import Pool
from scipy.sparse.csgraph import connected_components

class graphdataset(Dataset):
    
    def __init__(self,protein_coords,protein_types,pharm_coords,pharm_features,current_graph,pharm_pharm_radius=6,protein_pharm_radius=7,pool_processes=4,parallel=True,starter_points=None):
        super(graphdataset).__init__()
        self.pool_processes = pool_processes
        self.protein_coords = protein_coords
        self.protein_types = protein_types
        self.pharm_coords = pharm_coords
        self.pharm_features = pharm_features
        self.current_graph = current_graph
        self.protein_pharm_radius = protein_pharm_radius
        self.pharm_pharm_radius = pharm_pharm_radius
        self.parallel=parallel
        self.protein_pharm_distances = cdist(self.protein_coords,self.pharm_coords)
        self.datapoints = []
        
        if starter_points is not None:
            starter_points= np.where(starter_points)[0]#indices of starter points
            starter_coords=self.pharm_coords[starter_points]#coords of starter points
            if len(starter_coords.shape)==1:
                starter_coords=np.expand_dims(starter_coords,axis=0)
            pharm_pharm_distances = cdist(starter_coords,starter_coords)
            #find connected components based on edge distance threshold
            connected_graphs=connected_components(pharm_pharm_distances<self.pharm_pharm_radius)
            if connected_graphs[0]>1:
                print('Warning: starter points too far from each other, subset of starter points will be used')
            for i in range(connected_graphs[0]):
                starter_points_current=starter_points[connected_graphs[1]==i]
                starter_coords_current=starter_coords[connected_graphs[1]==i]
                starter_coords_current=starter_coords_current[0]
                if len(starter_coords_current.shape)==1:
                    starter_coords_current=np.expand_dims(starter_coords_current,axis=0)
                new_graph=HeteroData()
                new_graph=self.process_node_initial(starter_points_current[0],new_graph)
                for j in range(1,len(starter_points_current)):
                    new_graph_nodes = new_graph['pharm'].index
                    new_pharm_coords=self.pharm_coords[new_graph_nodes]
                    if len(new_pharm_coords.shape)==1:
                        new_pharm_coords=np.expand_dims(new_pharm_coords,axis=0)
                    pharm_pharm_distances = cdist(self.pharm_coords,new_pharm_coords)
                    new_graph=self.process_node(starter_points_current[j],new_graph,pharm_pharm_distances)
                self.datapoints.append(new_graph)
            for graph in self.datapoints:
                self.current_graph=graph
                self.datapoints=self.datapoints+self.extend_current_graph()
                self.datapoints.append(graph)
            self.current_graph=None
        else:
            self.datapoints=self.extend_current_graph() 
            if self.current_graph is not None:
                self.datapoints.append(self.current_graph)  

    def extend_current_graph(self):
        datapoints=[]
        if self.current_graph is None:
            self.current_graph=HeteroData()
            pharm_extension_nodes=range(self.pharm_coords.shape[0])
            datapoints  = self.generate_graphs(self.current_graph,pharm_extension_nodes)          
        else:
            current_graph_nodes = self.current_graph['pharm'].index
            current_pharm_coords=self.pharm_coords[current_graph_nodes]
            if len(current_pharm_coords.shape)==1:
                current_pharm_coords=np.expand_dims(current_pharm_coords,axis=0)
            pharm_pharm_distances = cdist(self.pharm_coords,current_pharm_coords)
            pharm_pharm_distances_min = pharm_pharm_distances.min(axis=1)
            pharm_extension_nodes = np.where(pharm_pharm_distances_min<self.pharm_pharm_radius)[0]
            datapoints = self.generate_graphs(self.current_graph,pharm_extension_nodes,pharm_pharm_distances)    
        return datapoints

        

    # def process_node_initial(self,i, provided_current_graph, pharm_features, pharm_coords, protein_pharm_distances, protein_pharm_radius, protein_types, protein_coords):
    #     current_graph = deepcopy(provided_current_graph)
    #     current_graph['pharm'].index = torch.tensor([i], dtype=torch.long)
    #     current_graph['pharm'].x = torch.tensor(np.expand_dims(pharm_features[i], axis=0), dtype=torch.float32)
    #     current_graph['pharm'].pos = torch.tensor(np.expand_dims(pharm_coords[i], axis=0), dtype=torch.float32)
    #     protein_nodes = np.where(protein_pharm_distances[:, i] < protein_pharm_radius)[0]
    #     if len(protein_nodes) == 0:
    #         return None
    #     current_graph['protein'].index = torch.tensor(protein_nodes, dtype=torch.long)
    #     current_graph['protein'].x = torch.tensor(protein_types[protein_nodes], dtype=torch.long)
    #     current_graph['protein'].pos = torch.tensor(protein_coords[protein_nodes], dtype=torch.float32)
    #     protein_nodes_graph = np.arange(len(protein_nodes))
    #     current_graph['protein', 'proteinpharm', 'pharm'].edge_index = torch.tensor(np.array([protein_nodes_graph, [0] * len(protein_nodes)]), dtype=torch.long)
    #     current_graph['protein', 'proteinpharm', 'pharm'].edge_attr = torch.tensor(protein_coords[protein_nodes] - pharm_coords[i], dtype=torch.float32)
    #     return current_graph  
    
    # def process_node(self,i, initial_graph, pharm_features, pharm_coords, protein_pharm_distances, protein_pharm_radius, protein_types, protein_coords,pharm_pharm_distances,pharm_pharm_radius):
    #     current_graph = deepcopy(initial_graph)

    #     if i in current_graph['pharm'].index:
    #         return None
        
    #     current_graph['pharm'].index=torch.cat((current_graph['pharm'].index,torch.tensor([i],dtype=torch.long)))
    #     current_graph['pharm'].x=torch.cat((current_graph['pharm'].x,torch.tensor(np.expand_dims(self.pharm_features[i],axis=0),dtype=torch.float32)))
    #     current_graph['pharm'].pos=torch.cat((current_graph['pharm'].pos,torch.tensor(np.expand_dims(self.pharm_coords[i],axis=0),dtype=torch.float32)))
    #     pharm_nodes=np.where(pharm_pharm_distances[i,:]<self.pharm_pharm_radius)[0]
    #     #pharm_nodes=current_graph['pharm'].index[pharm_nodes].tolist()
    #     try:
    #         current_graph['pharm','pharmpharm','pharm'].edge_index=torch.cat((current_graph['pharm','pharmpharm','pharm'].edge_index,torch.tensor(np.array([pharm_nodes,[len(current_graph['pharm'].index)-1]*len(pharm_nodes)]),dtype=torch.long)),dim=1)
    #         current_graph['pharm','pharmpharm','pharm'].edge_index=torch.cat((current_graph['pharm','pharmpharm','pharm'].edge_index,torch.tensor(np.array([[len(current_graph['pharm'].index)-1]*len(pharm_nodes),pharm_nodes]),dtype=torch.long)),dim=1)    
    #         current_graph['pharm','pharmpharm','pharm'].edge_attr=torch.cat((current_graph['pharm','pharmpharm','pharm'].edge_attr,torch.tensor(self.pharm_coords[pharm_nodes]-self.pharm_coords[i],dtype=torch.float32)))
    #         current_graph['pharm','pharmpharm','pharm'].edge_attr=torch.cat((current_graph['pharm','pharmpharm','pharm'].edge_attr,torch.tensor(self.pharm_coords[i]-self.pharm_coords[pharm_nodes],dtype=torch.float32)))
    #     except:
    #         current_graph['pharm','pharmpharm','pharm'].edge_index=torch.tensor(np.array([pharm_nodes,[len(current_graph['pharm'].index)-1]*len(pharm_nodes)]),dtype=torch.long)
    #         current_graph['pharm','pharmpharm','pharm'].edge_index=torch.cat((current_graph['pharm','pharmpharm','pharm'].edge_index,torch.tensor(np.array([[len(current_graph['pharm'].index)-1]*len(pharm_nodes),pharm_nodes]),dtype=torch.long)),dim=1)   
    #         current_graph['pharm','pharmpharm','pharm'].edge_attr=torch.tensor(self.pharm_coords[pharm_nodes]-self.pharm_coords[i],dtype=torch.float32)
    #         current_graph['pharm','pharmpharm','pharm'].edge_attr=torch.cat((current_graph['pharm','pharmpharm','pharm'].edge_attr,torch.tensor(self.pharm_coords[i]-self.pharm_coords[pharm_nodes],dtype=torch.float32)))
    #     protein_nodes=np.where(protein_pharm_distances[:,i]<self.protein_pharm_radius)[0]
    #     set1=set(protein_nodes)
    #     set2=set(current_graph['protein'].index.tolist())
    #     protein_nodes_intersection=list(set1&set2)
    #     if len(protein_nodes_intersection)>0:
    #         #protein_nodes_graph=current_graph['protein'].index.tolist().index(protein_nodes_intersection)
    #         protein_nodes_graph=[current_graph['protein'].index.tolist().index(x) for x in protein_nodes_intersection]
    #         current_graph['protein','proteinpharm','pharm'].edge_index=torch.cat((current_graph['protein','proteinpharm','pharm'].edge_index,torch.tensor(np.array([protein_nodes_graph,[len(current_graph['pharm'].index)-1]*len(protein_nodes_graph)]),dtype=torch.long)),dim=1)
    #         current_graph['protein','proteinpharm','pharm'].edge_attr=torch.cat((current_graph['protein','proteinpharm','pharm'].edge_attr,torch.tensor(self.protein_coords[protein_nodes_intersection]-self.pharm_coords[i],dtype=torch.float32)))
    #     protein_nodes_new=list(set1-set2)
    #     if len(protein_nodes_new)==0:
    #         return current_graph
    #     protein_nodes_graph=np.arange(len(current_graph['protein'].index.tolist()),len(current_graph['protein'].index.tolist())+len(protein_nodes_new))
    #     current_graph['protein','proteinpharm','pharm'].edge_index=torch.cat((current_graph['protein','proteinpharm','pharm'].edge_index,torch.tensor(np.array([protein_nodes_graph,[len(current_graph['pharm'].index)-1]*len(protein_nodes_graph)]),dtype=torch.long)),dim=1)
    #     current_graph['protein','proteinpharm','pharm'].edge_attr=torch.cat((current_graph['protein','proteinpharm','pharm'].edge_attr,torch.tensor(self.protein_coords[protein_nodes_new]-self.pharm_coords[i],dtype=torch.float32)))
    #     current_graph['protein'].index=torch.cat((current_graph['protein'].index,torch.tensor(protein_nodes_new,dtype=torch.long)))
    #     current_graph['protein'].x=torch.cat((current_graph['protein'].x,torch.tensor(self.protein_types[protein_nodes_new],dtype=torch.long)))
    #     current_graph['protein'].pos=torch.cat((current_graph['protein'].pos,torch.tensor(self.protein_coords[protein_nodes_new],dtype=torch.float32)))
    #     return current_graph

    def process_node_initial(self,i, provided_current_graph):
            current_graph = deepcopy(provided_current_graph)
            current_graph['pharm'].index = torch.tensor([i], dtype=torch.long)
            current_graph['pharm'].x = torch.tensor(np.expand_dims(self.pharm_features[i], axis=0), dtype=torch.float32)
            current_graph['pharm'].pos = torch.tensor(np.expand_dims(self.pharm_coords[i], axis=0), dtype=torch.float32)
            protein_nodes = np.where(self.protein_pharm_distances[:, i] < self.protein_pharm_radius)[0]
            if len(protein_nodes) == 0:
                return None
            current_graph['protein'].index = torch.tensor(protein_nodes, dtype=torch.long)
            current_graph['protein'].x = torch.tensor(self.protein_types[protein_nodes], dtype=torch.long)
            current_graph['protein'].pos = torch.tensor(self.protein_coords[protein_nodes], dtype=torch.float32)
            protein_nodes_graph = np.arange(len(protein_nodes))
            current_graph['protein', 'proteinpharm', 'pharm'].edge_index = torch.tensor(np.array([protein_nodes_graph, [0] * len(protein_nodes)]), dtype=torch.long)
            current_graph['protein', 'proteinpharm', 'pharm'].edge_attr = torch.tensor(self.protein_coords[protein_nodes] - self.pharm_coords[i], dtype=torch.float32)
            return current_graph  
    
    def process_node(self,i, initial_graph,pharm_pharm_distances):
                current_graph = deepcopy(initial_graph)

                if i in current_graph['pharm'].index:
                    return None
                
                current_graph['pharm'].index=torch.cat((current_graph['pharm'].index,torch.tensor([i],dtype=torch.long)))
                current_graph['pharm'].x=torch.cat((current_graph['pharm'].x,torch.tensor(np.expand_dims(self.pharm_features[i],axis=0),dtype=torch.float32)))
                current_graph['pharm'].pos=torch.cat((current_graph['pharm'].pos,torch.tensor(np.expand_dims(self.pharm_coords[i],axis=0),dtype=torch.float32)))
                pharm_nodes=np.where(pharm_pharm_distances[i,:]<self.pharm_pharm_radius)[0]
                #pharm_nodes=current_graph['pharm'].index[pharm_nodes].tolist()
                try:
                    current_graph['pharm','pharmpharm','pharm'].edge_index=torch.cat((current_graph['pharm','pharmpharm','pharm'].edge_index,torch.tensor(np.array([pharm_nodes,[len(current_graph['pharm'].index)-1]*len(pharm_nodes)]),dtype=torch.long)),dim=1)
                    current_graph['pharm','pharmpharm','pharm'].edge_index=torch.cat((current_graph['pharm','pharmpharm','pharm'].edge_index,torch.tensor(np.array([[len(current_graph['pharm'].index)-1]*len(pharm_nodes),pharm_nodes]),dtype=torch.long)),dim=1)    
                    current_graph['pharm','pharmpharm','pharm'].edge_attr=torch.cat((current_graph['pharm','pharmpharm','pharm'].edge_attr,torch.tensor(self.pharm_coords[pharm_nodes]-self.pharm_coords[i],dtype=torch.float32)))
                    current_graph['pharm','pharmpharm','pharm'].edge_attr=torch.cat((current_graph['pharm','pharmpharm','pharm'].edge_attr,torch.tensor(self.pharm_coords[i]-self.pharm_coords[pharm_nodes],dtype=torch.float32)))
                except:
                    current_graph['pharm','pharmpharm','pharm'].edge_index=torch.tensor(np.array([pharm_nodes,[len(current_graph['pharm'].index)-1]*len(pharm_nodes)]),dtype=torch.long)
                    current_graph['pharm','pharmpharm','pharm'].edge_index=torch.cat((current_graph['pharm','pharmpharm','pharm'].edge_index,torch.tensor(np.array([[len(current_graph['pharm'].index)-1]*len(pharm_nodes),pharm_nodes]),dtype=torch.long)),dim=1)   
                    current_graph['pharm','pharmpharm','pharm'].edge_attr=torch.tensor(self.pharm_coords[pharm_nodes]-self.pharm_coords[i],dtype=torch.float32)
                    current_graph['pharm','pharmpharm','pharm'].edge_attr=torch.cat((current_graph['pharm','pharmpharm','pharm'].edge_attr,torch.tensor(self.pharm_coords[i]-self.pharm_coords[pharm_nodes],dtype=torch.float32)))
                protein_nodes=np.where(self.protein_pharm_distances[:,i]<self.protein_pharm_radius)[0]
                set1=set(protein_nodes)
                set2=set(current_graph['protein'].index.tolist())
                protein_nodes_intersection=list(set1&set2)
                if len(protein_nodes_intersection)>0:
                    #protein_nodes_graph=current_graph['protein'].index.tolist().index(protein_nodes_intersection)
                    protein_nodes_graph=[current_graph['protein'].index.tolist().index(x) for x in protein_nodes_intersection]
                    current_graph['protein','proteinpharm','pharm'].edge_index=torch.cat((current_graph['protein','proteinpharm','pharm'].edge_index,torch.tensor(np.array([protein_nodes_graph,[len(current_graph['pharm'].index)-1]*len(protein_nodes_graph)]),dtype=torch.long)),dim=1)
                    current_graph['protein','proteinpharm','pharm'].edge_attr=torch.cat((current_graph['protein','proteinpharm','pharm'].edge_attr,torch.tensor(self.protein_coords[protein_nodes_intersection]-self.pharm_coords[i],dtype=torch.float32)))
                protein_nodes_new=list(set1-set2)
                if len(protein_nodes_new)==0:
                    return current_graph
                protein_nodes_graph=np.arange(len(current_graph['protein'].index.tolist()),len(current_graph['protein'].index.tolist())+len(protein_nodes_new))
                current_graph['protein','proteinpharm','pharm'].edge_index=torch.cat((current_graph['protein','proteinpharm','pharm'].edge_index,torch.tensor(np.array([protein_nodes_graph,[len(current_graph['pharm'].index)-1]*len(protein_nodes_graph)]),dtype=torch.long)),dim=1)
                current_graph['protein','proteinpharm','pharm'].edge_attr=torch.cat((current_graph['protein','proteinpharm','pharm'].edge_attr,torch.tensor(self.protein_coords[protein_nodes_new]-self.pharm_coords[i],dtype=torch.float32)))
                current_graph['protein'].index=torch.cat((current_graph['protein'].index,torch.tensor(protein_nodes_new,dtype=torch.long)))
                current_graph['protein'].x=torch.cat((current_graph['protein'].x,torch.tensor(self.protein_types[protein_nodes_new],dtype=torch.long)))
                current_graph['protein'].pos=torch.cat((current_graph['protein'].pos,torch.tensor(self.protein_coords[protein_nodes_new],dtype=torch.float32)))
                return current_graph

    def generate_graphs(self,current_graph,pharm_extension_nodes,pharm_pharm_distances=None):
        #TODO ensure directionality of edges is correct, reverse edges
        if pharm_pharm_distances is None:
            if self.parallel:
                with Pool(processes=self.pool_processes) as pool:  # You can adjust the number of processes as needed
                    args = [(i, self.current_graph) for i in pharm_extension_nodes]
                    datapoints = pool.starmap(self.process_node_initial, args)
                datapoints = [graph for graph in datapoints if graph is not None]
            else:
                datapoints=[]
                for i in pharm_extension_nodes:
                    current_graph=self.process_node_initial(i, self.current_graph)
                    if current_graph is not None:
                        datapoints.append(current_graph)

        else:
            if self.parallel:
                with Pool(processes=self.pool_processes) as pool:  # You can adjust the number of processes as needed
                    args = [(i, self.current_graph,pharm_pharm_distances) for i in pharm_extension_nodes]
                    modified_graphs = pool.starmap(self.process_node, args)
                
                datapoints = [graph for graph in modified_graphs if graph is not None]
            else:
                datapoints=[]
                for i in pharm_extension_nodes:
                    current_graph=self.process_node(i, self.current_graph,pharm_pharm_distances)
                    if current_graph is not None:
                        datapoints.append(current_graph)
            
        return datapoints
    
    def __len__(self):
        return len(self.datapoints)
    
    def __getitem__(self,idx):
        return self.datapoints[idx]
    
    def get(self,idx):
        return self.datapoints[idx]
    
    def pop(self,idx):
        self.datapoints.pop(idx)

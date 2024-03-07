'''Se(3) heterogenous graph neural network using pytorch geometric and e3nn library.'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter
from e3nn import o3
from e3nn.nn import BatchNorm
import torch_geometric.nn as pyg_nn
from torch_geometric.nn import global_mean_pool
from torch_geometric.nn import HeteroConv
import torch_geometric.transforms as T

#adapted from torsional diffusion network - https://github.com/gcorso/torsional-diffusion/blob/master/diffusion/score_model.py

class TensorProductConvLayer(torch.nn.Module):
    def __init__(self, in_irreps, sh_irreps, out_irreps, n_edge_features, residual=True, batch_norm=True):
        super(TensorProductConvLayer, self).__init__()
        self.in_irreps = in_irreps
        self.out_irreps = out_irreps
        self.sh_irreps = sh_irreps
        self.residual = residual

        self.tp = tp = o3.FullyConnectedTensorProduct(in_irreps, sh_irreps, out_irreps, shared_weights=False)

        self.fc = nn.Sequential(
            nn.Linear(n_edge_features, n_edge_features),
            nn.ReLU(),
            nn.Linear(n_edge_features, tp.weight_numel)
        )
        self.batch_norm = BatchNorm(out_irreps) if batch_norm else None

    def forward(self, node_attr, edge_index, edge_attr, edge_sh, out_nodes=None, reduce='sum'):

        edge_src, edge_dst = edge_index
        if type(node_attr) is tuple:
            tp = self.tp(node_attr[0][edge_src], edge_sh, self.fc(edge_attr))
        else:
            tp = self.tp(node_attr[edge_src], edge_sh, self.fc(edge_attr))
        if type(node_attr) is tuple:
            out_nodes = out_nodes or node_attr[1].shape[0]
        else:
            out_nodes = out_nodes or node_attr.shape[0]
        out = scatter(tp, edge_dst, dim=0, dim_size=out_nodes, reduce=reduce)
        if self.residual:
            if type(node_attr) is tuple:
                padded = F.pad(node_attr[1], (0, out.shape[-1] - node_attr[1].shape[-1]))
            else:
                padded = F.pad(node_attr, (0, out.shape[-1] - node_attr.shape[-1]))
            out = out + padded

        if self.batch_norm:
            out = self.batch_norm(out)

        return out

class GaussianSmearing(torch.nn.Module):
    def __init__(self, start=0.0, stop=5.0, num_gaussians=50):
        super().__init__()
        offset = torch.linspace(start, stop, num_gaussians)
        self.coeff = -0.5 / (offset[1] - offset[0]).item() ** 2
        self.register_buffer('offset', offset)

    def forward(self, dist):
        dist=dist.unsqueeze(-1)
        dist=dist.repeat(1,self.offset.shape[0])
        dist = dist - self.offset.view(1, -1)
        dist_embed=torch.exp(self.coeff * torch.pow(dist, 2))
        return dist_embed


class Se3NN(torch.nn.Module):

    def __init__(self,in_pharm_node_features=32,in_prot_node_features=14,sh_lmax=2,ns=32,nv=8,num_conv_layers=4,max_radius=6,radius_embed_dim=50,batch_norm=True,residual=True):
        super(Se3NN, self).__init__()
        self.in_pharm_node_features = in_pharm_node_features
        self.in_prot_node_features = in_prot_node_features
        self.sh_lmax = sh_lmax
        self.ns = ns
        self.nv = nv
        self.num_conv_layers = num_conv_layers
        self.max_radius = max_radius
        self.radius_embed_dim = radius_embed_dim
        self.batch_norm = batch_norm
        self.residual = residual

        self.protein_embedding = nn.Sequential(
            torch.nn.Embedding(in_prot_node_features, ns),
            nn.ReLU(),
            nn.Linear(ns, ns))
        
        self.pharmacophore_embedding = nn.Sequential(
            nn.Linear(in_pharm_node_features, ns),
            nn.ReLU(),
            nn.Linear(ns, ns))
        
        self.pharm_pharm_edge_embedding = nn.Sequential(
            nn.Linear(radius_embed_dim,ns),
            nn.ReLU(),
            nn.Linear(ns,ns))
        
        self.prot_pharm_edge_embedding = nn.Sequential(
            nn.Linear(radius_embed_dim,ns),
            nn.ReLU(),
            nn.Linear(ns,ns))
        
        self.distance_expansion = GaussianSmearing(0.0, max_radius, radius_embed_dim)
        
        self.sh_irreps = o3.Irreps.spherical_harmonics(lmax=sh_lmax)

        if sh_lmax==2:
             self.irrep_seq = [
                f'{ns}x0e',
                f'{ns}x0e + {nv}x1o + {nv}x2e',
                f'{ns}x0e + {nv}x1o + {nv}x2e + {nv}x1e + {nv}x2o',
                f'{ns}x0e + {nv}x1o + {nv}x2e + {nv}x1e + {nv}x2o + {ns}x0o',
                f'{ns}x0e'
            ]
        else:
            self.irrep_seq=[
                f'{ns}x0e',
                f'{ns}x0e + {nv}x1o',
                f'{ns}x0e + {nv}x1o + {nv}x1e',
                f'{ns}x0e + {nv}x1o + {nv}x1e + {ns}x0o',
                f'{ns}x0e'
            ]

        conv_layers = []
        for i in range(num_conv_layers):
            in_irreps = self.irrep_seq[min(i, len(self.irrep_seq) - 1)]
            out_irreps = self.irrep_seq[min(i + 1, len(self.irrep_seq) - 1)]
            layer = HeteroConv({('protein','proteinpharm','pharm'):TensorProductConvLayer(
                in_irreps=in_irreps,
                sh_irreps=self.sh_irreps,
                out_irreps=out_irreps,
                n_edge_features=3 * ns,
                residual=residual,
                batch_norm=batch_norm
            ),('pharm','rev_proteinpharm','protein'):TensorProductConvLayer(
                in_irreps=in_irreps,
                sh_irreps=self.sh_irreps,
                out_irreps=out_irreps,
                n_edge_features=3 * ns,
                residual=residual,
                batch_norm=batch_norm
            ),('pharm','pharmpharm','pharm'):TensorProductConvLayer(
                in_irreps=in_irreps,
                sh_irreps=self.sh_irreps,
                out_irreps=out_irreps,
                n_edge_features=3 * ns,
                residual=residual,
                batch_norm=batch_norm
            )},aggr='sum')
            conv_layers.append(layer)
        self.conv_layers = nn.ModuleList(conv_layers)
        
        self.final_linear = nn.Sequential(
            nn.Linear(self.ns, self.ns, bias=False),
            nn.ReLU(),
            nn.Linear(self.ns, 1, bias=False)
        )
    
    def forward(self,data):
        _,edge_types=data.metadata()
        #pharmpharm edge attr info gets lost in ToUndirected(), so we need to save it
        if ('pharm','pharmpharm','pharm') in edge_types:
            pharm_pharm_edge_attr=data['pharm','pharmpharm','pharm'].edge_attr.clone()
            pharm_pharm_edge_index=data['pharm','pharmpharm','pharm'].edge_index.clone()
        data = T.ToUndirected()(data)
        if ('pharm','pharmpharm','pharm') in edge_types:
            assert (data['pharm','pharmpharm','pharm'].edge_index.shape==pharm_pharm_edge_index.shape)
        x_prot = data['protein'].x.squeeze(-1)
        x_pharm = data['pharm'].x
        #x_dict['protein'] = self.protein_embedding(x_prot)
        data['protein'].x = self.protein_embedding(x_prot)

        #x_dict['pharma'] = self.pharmacophore_embedding(x_pharm)
        data['pharm'].x = self.pharmacophore_embedding(x_pharm)
       
        #prot_pharm_src,prot_pharm_dst=edge_index_dict['protein','proteinpharm','pharm']
        prot_pharm_src,prot_pharm_dst=data['protein','proteinpharm','pharm'].edge_index
        #prot_pharm_edge_attr = edge_attr_dict['protein','proteinpharm','pharm']
        prot_pharm_edge_attr = data['protein','proteinpharm','pharm'].edge_attr

        pharm_prot_src,pharm_prot_dst=data['pharm','rev_proteinpharm','protein'].edge_index
        data['pharm','rev_proteinpharm','protein'].edge_attr = -data['pharm','rev_proteinpharm','protein'].edge_attr
        pharm_prot_edge_attr = data['pharm','rev_proteinpharm','protein'].edge_attr
        
        #edge_sh={}
        data['protein','proteinpharm','pharm'].edge_sh =  o3.spherical_harmonics(self.sh_irreps, prot_pharm_edge_attr, normalize=True, normalization='component')
        data['pharm','rev_proteinpharm','protein'].edge_sh =  o3.spherical_harmonics(self.sh_irreps, pharm_prot_edge_attr, normalize=True, normalization='component')
        #edge_sh['proteinpharm']=o3.spherical_harmonics(self.sh_irreps, prot_pharm_edge_attr, normalize=True, normalization='component')
        #edge_attr_dict['prot','proteinpharm','pharm'] = self.prot_pharm_edge_embedding(self.distance_expansion(prot_pharm_edge_distance))
        prot_pharm_edge_distance=prot_pharm_edge_attr.norm(dim=1)
        pharm_prot_edge_distance=pharm_prot_edge_attr.norm(dim=1)
        data['protein','proteinpharm','pharm'].edge_distance_embed = self.prot_pharm_edge_embedding(self.distance_expansion(prot_pharm_edge_distance))
        data['pharm','rev_proteinpharm','protein'].edge_distance_embed = self.prot_pharm_edge_embedding(self.distance_expansion(pharm_prot_edge_distance))
        if ('pharm','pharmpharm','pharm') in edge_types:
            #pharm_pharm_src,pharm_pharm_dst=edge_index_dict['pharm','pharmpharm','pharm']
            data['pharm','pharmpharm','pharm'].edge_index=pharm_pharm_edge_index
            pharm_pharm_src,pharm_pharm_dst=data['pharm','pharmpharm','pharm'].edge_index
            #pharm_pharm_edge_attr = edge_attr_dict['pharm','pharmpharm','pharm']
            data['pharm','pharmpharm','pharm'].edge_attr=pharm_pharm_edge_attr
            data['pharm','pharmpharm','pharm'].edge_sh =  o3.spherical_harmonics(self.sh_irreps, pharm_pharm_edge_attr, normalize=True, normalization='component')
            pharm_pharm_edge_distance=pharm_pharm_edge_attr.norm(dim=1)
            #edge_attr_dict['pharm','pharmpharm','pharm']=self.pharm_pharm_edge_embedding(self.distance_expansion(pharm_pharm_edge_distance))
            data['pharm','pharmpharm','pharm'].edge_distance_embed=self.pharm_pharm_edge_embedding(self.distance_expansion(pharm_pharm_edge_distance))    
        for layer in self.conv_layers:
            #edge_attr_dict['pharm','pharmpharm','pharm']= torch.cat([edge_attr_dict['pharm','pharmpharm','pharm'],x_dict['pharma'][pharm_pharm_src,:self.ns],x_dict['pharma'][pharm_pharm_dst,:self.ns]],dim=1)
            data['protein','proteinpharm','pharm'].edge_attr= torch.cat([data['protein','proteinpharm','pharm'].edge_distance_embed,data['protein'].x[prot_pharm_src,:self.ns],data['pharm'].x[prot_pharm_dst,:self.ns]],dim=1)
            data['pharm','rev_proteinpharm','protein'].edge_attr= torch.cat([data['pharm','rev_proteinpharm','protein'].edge_distance_embed,data['pharm'].x[pharm_prot_src,:self.ns],data['protein'].x[pharm_prot_dst,:self.ns]],dim=1)
            if ('pharm','pharmpharm','pharm') in edge_types:
                data['pharm','pharmpharm','pharm'].edge_attr= torch.cat([data['pharm','pharmpharm','pharm'].edge_distance_embed,data['pharm'].x[pharm_pharm_src,:self.ns],data['pharm'].x[pharm_pharm_dst,:self.ns]],dim=1)
            data.x_dict= layer(data.x_dict,data.edge_index_dict,data.edge_attr_dict,data.edge_sh_dict)
        
        x = data['pharm'].x
        x = global_mean_pool(x, data['pharm'].batch)

        x = self.final_linear(x)
        return x






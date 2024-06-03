import sys
sys.path.append('./Pharmnn')
from se3nn import Se3NN
import torch
import argparse
import json
import pandas as pd
from Pharmnn.dataset import Inference_Dataset
try:
    import molgrid.openbabel as ob
except ImportError:
    import openbabel as ob
try:
    from molgrid.openbabel import pybel
except ImportError:
    from openbabel import openbabel
    from openbabel import pybel
import molgrid
from molgrid import CoordinateSet
from rdkit import Chem
from rdkit.Chem import rdmolfiles
from environment import Inference_environment
from Pharmnn.inference import infer
from Pharmnn.pharm_rec import get_mol_pharm
from qlearning import select_action
from se3nn import Se3NN
import os
import pickle as pkl
import json

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def parse_arguments():
    parser = argparse.ArgumentParser()
    #general run parameters
    parser.add_argument('--top_dir', type=str, default='')
    parser.add_argument('--receptor', type=str, default='',help='receptor information, if not given then taken from input json')
    parser.add_argument('--ligand', type=str, default='',help='ligand information, if not provided then binding site needs to be provided')
    parser.add_argument('--x_center', type=float, default=None)
    parser.add_argument('--y_center', type=float, default=None)
    parser.add_argument('--z_center', type=float, default=None)
    parser.add_argument('--x_size', type=float, default=None)
    parser.add_argument('--y_size', type=float, default=None)
    parser.add_argument('--z_size', type=float, default=None)
    parser.add_argument('--input_json', type=str, default='')
    parser.add_argument('--starter_json', type=str, default='None',help='starter pharmacophore points')
    parser.add_argument('--features', default='combine', choices=['combine','cnn_only','ligand_only'], help='The type of pharmacophore features to use')
    parser.add_argument('--pharmnn_session', type=str, default=None,help='reload previous pharmnn features and points')
    parser.add_argument('--dump_pharmnn', type=str, default=None,help='pharmnn dump file name')
    parser.add_argument('--output_prefix', type=str, default='pharmnn',help='prefix for output json session files')

    #cnn_feature_points_parameters
    parser.add_argument('--cnn_batch_size',default=512,type=int,help='cnn batch size')
    parser.add_argument('--grid_dimension',default=9.5,type=float,help='dimension in angstroms of grid; only 5 is supported with gist')
    parser.add_argument('--dataset_threshold',default=0.9,type=float,help='greedy threshold at which to sample points for active learning')
    parser.add_argument('--rotate', type=int,default=0,help='random rotations of pdb grid')
    parser.add_argument('--use_se3', type=int,default=0,help='use se3 convolutions')
    parser.add_argument('--seed',default=42,type=int,help='random seed')
    parser.add_argument('--autobox_extend',default=4,type=int,help='amount to expand autobox by')
    parser.add_argument('--create_dx',help="Create dx files of the predictions",action='store_true')
    parser.add_argument('--prefix_dx',type=str,default="",help='prefix for dx files')
    parser.add_argument('--cnn_output_pred',help="output predictions into a text file",action='store_true')
    parser.add_argument('--cnn_round_pred',help="round up predictions in dx files",action='store_true')
    parser.add_argument('--cnn_output',type=str,help='output file of the predictions')
    parser.add_argument('--verbose',help="verbse complex working on",action='store_true')
    parser.add_argument('--cnn_prefix_xyz',type=str,default="",help='prefix for xyz files')
    parser.add_argument('--prob_threshold',default=0.9,type=float,help='probability threshold for masking')
    parser.add_argument('--clus_threshold',default=1.5,type=float,help='distance threshold for clustering pharmacophore points')
    parser.add_argument('--xyz_rank',default=0,type=int,help='output only top ranked xyz points')
    parser.add_argument('--category_wise',help="output top ranked for each pharmacophore category",action='store_true')
    parser.add_argument('--density_score',help="output top ranked for each pharmacophore category by density",action='store_true')
    parser.add_argument('--density_distance_threshold',default=2.0,type=float,help='distance threshold for density score')

    #RL parameters
    parser.add_argument('--max_size',default=10,type=int,help='maximum size of pharmacophore')
    parser.add_argument('--batch_size',default=50,type=int,help='RL batch size')
    parser.add_argument('--pharm_pharm_radius',default=12,type=int,help='pharmacophore-pharmacophore edge threshold')
    parser.add_argument('--protein_pharm_radius',default=11,type=int,help='protein-pharmacophore edge threshold')
    parser.add_argument('--in_pharm_node_features',type=int,default=32)
    parser.add_argument('--in_prot_node_features',type=int,default=14)
    parser.add_argument('--sh_lmax',type=int,default=2)
    parser.add_argument('--ns',type=int,default=32)
    parser.add_argument('--nv',type=int,default=8)
    parser.add_argument('--num_conv_layers',type=int,default=6)
    parser.add_argument('--max_radius',type=float,default=9)
    parser.add_argument('--radius_embed_dim',type=int,default=78)
    parser.add_argument('--batch_norm',type=bool,default=True)
    parser.add_argument('--residual',type=bool,default=True)
    parser.add_argument('--min_features',type=int,default=3,help='minimum number of features in predicted pharmacophore')
    parser.add_argument('--max_features',type=int,default=10,help='maximum number of features in predicted pharmacophore')
    parser.add_argument('--parallel',action='store_true',default=False,help='use parallel processing for generating datasets')
    parser.add_argument('--pool_processes',type=int,default=1,help='number of processes to use for parallel processing')
    args = parser.parse_args()
    return args


def cnn_inference_arguments(args):
    args_inference=argparse.Namespace()
    for arg in vars(args):
        setattr(args_inference,arg,getattr(args,arg))
    setattr(args_inference,'train_data','')
    setattr(args_inference,'test_data','')
    setattr(args_inference,'create_dataset',False)
    setattr(args_inference,'batch_size',getattr(args,'cnn_batch_size'))
    setattr(args_inference,'expand_width',0)
    setattr(args_inference,'use_gist',0)
    setattr(args_inference,'use_se3',0)
    setattr(args_inference,'model','./Pharmnn/models/obabel_chemsplit1_2_best_model.pkl')
    setattr(args_inference,'negative_output','')
    setattr(args_inference,'output',getattr(args,'cnn_output'))
    setattr(args_inference,'output_pred',getattr(args,'cnn_output_pred'))
    setattr(args_inference,'round_pred',getattr(args,'cnn_round_pred'))
    setattr(args_inference,'spherical',True)
    setattr(args_inference,'xyz',True)
    setattr(args_inference,'prefix_xyz',getattr(args,'cnn_prefix_xyz'))
    return args_inference


    

def extract_json(json_file):
    """Takes a json files as input and returns a pandas dataframe with the headers as the keys of the json file"""
    with open(json_file) as f:
        data = json.load(f)
    return data

def dict_to_df(dict,vector=False):
    features=[]
    x_values=[]
    y_values=[]
    z_values=[]
    vector_values=[]
    svector_values=[]
    for key, positions in dict.items():
        for position in positions:
            features.append(key)
            x_values.append(position[0])
            y_values.append(position[1])
            z_values.append(position[2])
            vector_values.append(None)
            svector_values.append(None)
    # Create a Pandas DataFrame
    if vector:
        df = pd.DataFrame({
            'Feature': features,
            'x': x_values,
            'y': y_values,
            'z': z_values,
            'vector': vector_values,
            'svector': svector_values
        })
    else:
        df = pd.DataFrame({
            'Feature': features,
            'x': x_values,
            'y': y_values,
            'z': z_values
        })
    return df


def pharm_rec_df(rdmol,obmol):
    pharm_rec_features=get_mol_pharm(rdmol,obmol)
    df=dict_to_df(pharm_rec_features)
    return df

def points_to_df(points):
    if len(points)==0:
        return None
    points_df=pd.DataFrame(points)
    points_df=points_df[points_df['enabled']==True]
    if not 'vector' in points_df.columns:
        points_df['vector']=None
    if not 'svector' in points_df.columns:
        points_df['svector']=None
    #take only name,x,y,z and vector and svector columns
    points_df=points_df[['name','x','y','z','vector','svector']]
    #rename name column as Feature
    points_df=points_df.rename(columns={'name':'Feature'})
    #drop comlumns where Feature is InclusionSphere
    points_df=points_df[points_df.Feature != 'InclusionSphere']
    #reset index
    points_df=points_df.reset_index(drop=True)
    return points_df

def get_rdmol_obmol(file):
    file_suffix=file.split('.')[-1]
    if file_suffix=='pdb':
        rdmol=rdmolfiles.MolFromPDBFile(file,sanitize=False,proximityBonding=False)
    elif file_suffix=='mol2':
        rdmol=rdmolfiles.MolFromMol2File(file,sanitize=False,cleanupSubstructures=False)
    elif file_suffix=='mol':
        rdmol=rdmolfiles.MolFromMolFile(file,sanitize=False,strictParsing=False)
    elif file_suffix=='sdf':
        rdmol=rdmolfiles.SDMolSupplier(file,sanitize=False,strictParsing=False)
        rdmol=rdmol[0]
    obmol =next(pybel.readfile(file_suffix, file))
    return rdmol,obmol

def main(args):

    starter_points_df=None
    points_df=None
    if len(args.input_json)>0:
        input_json=extract_json(args.input_json)
    else:
        input_json=None
    receptor=args.receptor
    ligand=args.ligand
    s = molgrid.ExampleProviderSettings(data_root=args.top_dir)
    coord_reader = molgrid.CoordCache(molgrid.defaultGninaReceptorTyper,s)
    receptor_file_generated=False
    ligand_file_generated=False
    json_file_generated=False
    ligand_string=None
    ligand_file_name_env=None
    
    if args.verbose:
        print('reading receptor')

    if len(receptor)==0:
        if input_json is None:
            raise ValueError('No receptor provided')
        receptor_string=input_json['receptor']
        receptor_file_name=input_json['recname']
        receptor_file_suffix=receptor_file_name.split('.')[-1]
        #an obscure name so we dont overwrite any files
        receptor_file=open('ihopethisisnotafilename.'+receptor_file_suffix,'w')
        receptor_file.write(receptor_string)
        receptor_file.close()
        receptor_file_generated=True
        receptor_rdmol,receptor_pybel=get_rdmol_obmol('ihopethisisnotafilename.'+receptor_file_suffix)
        receptor=coord_reader.make_coords('ihopethisisnotafilename.'+receptor_file_suffix)
        receptor_file_name='ihopethisisnotafilename.'+receptor_file_suffix
        
    else:
        receptor_file_name=args.receptor
        receptor_rdmol,receptor_pybel=get_rdmol_obmol(receptor_file_name)
        receptor_string=open(receptor_file_name).read()
        receptor=coord_reader.make_coords(receptor_file_name)
    
    if args.verbose:
        print('reading starter pharmacophore')
    
    if args.starter_json!='None':
        starter_json=extract_json(args.starter_json)
        starter_points_df=points_to_df(starter_json['points'])
        
    if args.verbose:
        print('defining binding site')

    
    if args.pharmnn_session is None:
        if len(args.ligand)>0:
            ligand_file_name=args.ligand
            ligand_string=open(ligand_file_name).read()
            ligand_file_name_env=ligand_file_name
            ligand_file_suffix=ligand_file_name.split('.')[-1]
            ligand_pybel=next(pybel.readfile(ligand_file_suffix, ligand_file_name))
            ligand=coord_reader.make_coords(ligand_file_name)
        elif input_json is not None and 'ligand' in input_json.keys():
            ligand_string=input_json['ligand']
            ligand_file_name=input_json['ligandFormat']
            ligand_file_suffix=ligand_file_name.split('.')[-1]
            #an obscure name so we dont overwrite any files
            ligand_file=open('ihopethisisnotafilename_ligand.'+ligand_file_suffix,'w')
            ligand_file.write(ligand_string)
            ligand_file.close()
            ligand_file_generated=True
            ligand_file_name=ligand_file_name_env='ihopethisisnotafilename_ligand.'+ligand_file_suffix
            ligand_file_suffix=ligand_file_name.split('.')[-1]
            ligand_pybel=next(pybel.readfile(ligand_file_suffix, ligand_file_name))
            ligand=coord_reader.make_coords(ligand_file_name)
        else:
            if args.x_center is None or args.y_center is None or args.z_center is None or args.x_size is None or args.y_size is None or args.z_size is None:
                raise ValueError('No binding site provided')
            else:
                x_min=args.x_center-args.x_size/2
                x_max=args.x_center+args.x_size/2
                y_min=args.y_center-args.y_size/2
                y_max=args.y_center+args.y_size/2
                z_min=args.z_center-args.z_size/2
                z_max=args.z_center+args.z_size/2
                ligand_obmol=ob.OBMol()
                ligand_pybel=pybel.Molecule(ligand_obmol)
                atom_1 = ob.OBAtom()
                atom_1.SetAtomicNum(7)  # Set the atomic number (e.g., carbon)
                atom_1.SetVector(x_min, y_min, z_min)  # Set 3D coordinates (x, y, z)
                atom_2 = ob.OBAtom()
                atom_2.SetAtomicNum(7)  # Set the atomic number (e.g., carbon)
                atom_2.SetVector(x_max, y_max, z_max)  # Set 3D coordinates (x, y, z)
                ligand_pybel.OBMol.AddAtom(atom_1)
                ligand_pybel.OBMol.AddAtom(atom_2)
                setattr(args,'autobox_extend',0)
                ligand_file_name='ihopethisisnotafilename_ligand.pdb'
                ligand_pybel.write('pdb',ligand_file_name,overwrite=True)
                ligand=coord_reader.make_coords(ligand_file_name)
                
        
        dataset=Inference_Dataset(receptor,ligand,auto_box_extend=args.autobox_extend,grid_dimension=args.grid_dimension,rotate=args.rotate,starter_df=starter_points_df)
        
        pharm_rec_features=pharm_rec_df(receptor_rdmol,receptor_pybel)

        #get the pharmacophore points

        if 'combine' in args.features or 'ligand_only' in args.features:
            if len(args.ligand)>0:
                if not os.path.isfile('pharmit'):
                    os.system('wget --no-check-certificate https://github.com/dkoes/pharmit/releases/download/v1.0/pharmit')
                    os.system('chmod +x pharmit')
                os.system(f'./pharmit pharma -receptor {receptor_file_name} -in {ligand_file_name} -out ihopethisisnotafile.json')
                ligand_points_df=points_to_df(extract_json('ihopethisisnotafile.json')['points'])
                json_file_generated=True
                dataset.add_points(ligand_points_df)
            else:
                if input_json is not None:
                    ligand_points_df=points_to_df(input_json['points'])
                    if ligand_points_df is not None:
                        dataset.add_points(ligand_points_df)
                    else:
                        raise ValueError('No ligand pharmacophore feature provided or detected')
                
                else:
                    raise ValueError('No ligand pharmacophore feature provided or detected')


        if 'combine' in args.features or 'cnn_only' in args.features:
            if args.verbose:
                print('predicting pharmacophore feature points')
            
            args_cnn_inference=cnn_inference_arguments(args)
            returned_lists=infer(args_cnn_inference,dataset,pharm_rec_features)
            predicted_feature_points=returned_lists[0][0][0]
            predicted_feature_points_df=dict_to_df(predicted_feature_points,vector=True)
            dataset.add_points(predicted_feature_points_df)
        
        if args.verbose:
            print('added pharmacophore points to dataset')
        
        #check if there are enough points to form a pharmacophore
        if dataset.points is None or len(dataset.points)<3:
            raise ValueError('Not enough pharmacophore points to form a pharmacophore')
            

        if args.verbose:
            print('getting cnn hidden features')
        #get the hidden features
        dataloader=torch.utils.data.DataLoader(dataset,batch_size=args.cnn_batch_size,shuffle=False,num_workers=0,drop_last=False)
        cnn_net= torch.load('./Pharmnn/models/obabel_chemsplit1_2_best_model.pkl')
        cnn_net.eval()

        for i,batch in enumerate(dataloader):
            inputs = batch['grid'].to(device)
            with torch.no_grad():
                output,hidden_features=cnn_net(inputs)
                if i==0:
                    cnn_hidden_features=hidden_features
                else:
                    cnn_hidden_features=torch.cat((cnn_hidden_features,hidden_features),0)
        
        assert cnn_hidden_features.shape[0]==dataset.__len__()

        feature_points=dataset.get_points()
        if args.dump_pharmnn is not None:
            pkl.dump([feature_points,cnn_hidden_features],open(args.dump_pharmnn,'wb'))
    else:
        if args.verbose:
            print('loading pharmnn session')
        feature_points,cnn_hidden_features=pkl.load(open(args.pharmnn_session,'rb'))
        if args.verbose:
            print('pharmnn session loaded')
    
    if args.verbose:
        print('predicting pharmacophores')
    #predict the pharmacophores
    pharm_env=Inference_environment(receptor,receptor_string,receptor_file_name,ligand_string,ligand_file_name_env,feature_points,cnn_hidden_features,args.batch_size,args.top_dir,args.pharm_pharm_radius,args.protein_pharm_radius,args.max_size,args.parallel,args.pool_processes)

    if 'ligand_only' in args.features:
        models=['models/model_ligand.pt','models/model_cnn_1.pt','models/model_cnn_2.pt','models/model_cnn_3.pt','models/model_cnn_4.pt','models/model_cnn_5.pt']
    else:
        models=['models/model_cnn_1.pt','models/model_cnn_2.pt','models/model_cnn_3.pt','models/model_cnn_4.pt','models/model_cnn_5.pt']
    for model in models:
        policy_net = Se3NN(in_pharm_node_features=args.in_pharm_node_features, in_prot_node_features=args.in_prot_node_features, sh_lmax=args.sh_lmax, ns=args.ns, nv=args.nv, num_conv_layers=args.num_conv_layers, max_radius=args.max_radius, radius_embed_dim=args.radius_embed_dim, batch_norm=args.batch_norm, residual=args.residual).to(device)
        try:
            policy_net.load_state_dict(torch.load(model))
        except:
            new_state_dict ={}
            state_dict = torch.load(model)
            for key in state_dict.keys():
                if 'convs' in key:
                    key_list=key.split('.')
                    key_sub_list=key_list[3].split('__')
                    key_list[3]='___'.join(key_sub_list)
                    key_list[3]='<'+key_list[3]+'>'
                    new_key='.'.join(key_list)
                    new_state_dict[new_key]=state_dict[key]
                else:
                    new_state_dict[key]=state_dict[key]
            policy_net.load_state_dict(new_state_dict)
        policy_net.eval()
        state_loader=pharm_env.reset()
        state=None
        done=False
        steps_done=0
        while not done:
            next_state,steps_done,value = select_action(state,state_loader,policy_net,epsilon_start=0,epsilon_end=0,epsilon_decay=0,steps_done=steps_done,test=True,min_features=args.min_features) 
            next_state_dataloader,done=pharm_env.step(next_state,steps_done)
            state=next_state
            state_loader=next_state_dataloader
        try: #if json_dict already exists
            json_dict_new=pharm_env.state_to_json(state,min_features=args.min_features,label=model.split('/')[1].split('.')[0])
            json_dict["points"].extend(json_dict_new["points"])
        except:
            json_dict=pharm_env.state_to_json(state,min_features=args.min_features,label=model.split('/')[1].split('.')[0])
        json.dump(json_dict,open(args.output_prefix+'_predicted_pharmacophores.json','w'))   
    if receptor_file_generated:
        os.remove(receptor_file_name)
    if ligand_file_generated:
        os.remove(ligand_file_name)
    if json_file_generated:
        os.remove('ihopethisisnotafile.json')

if __name__ == '__main__':

    args=parse_arguments()
    main(args)



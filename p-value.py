import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import random
import argparse

parser = argparse.ArgumentParser(description='obtain p-value')
parser.add_argument('--file', type=str, default='../Pharmnn/data/test_pharmrl_dataset.txt', help='file name')
parser.add_argument('--upperbound', type=int, default=8, help='largest size of graph + 1')
parser.add_argument('--num_episodes', type=int, default=100000, help='number of episodes')
parser.add_argument('--f1_score', type=float, default=0.45, help='f1 score for p-value')

args = parser.parse_args()

df=pd.read_csv(args.file, sep=';', header=None)
unique_values=df.iloc[:,4].unique()

target_list=[]
for name in unique_values:
    target_list.append(name.split('/')[-2])

def get_mean(target_list,upperbound,normalize=True):
    mean_f1=[]
    num_enumerate=1
    for target in target_list:
        target_df=pickle.load(open('../Pharmnn/data/dude/all/'+target+'/target_df.pkl','rb'))
        if normalize:
            target_df['f1']=target_df['f1']/target_df['f1'].max()
        mean_f1_target=[]
        for num in range(3,upperbound):
            target_df_num=target_df[target_df['file'].str.startswith('pharmit_'+str(num)+'_')==1].sort_values(by='f1',ascending=False)
            #remove duplicate rows on file
            target_df_num=target_df_num.drop_duplicates(subset='file')
            mean_f1_target+=target_df_num['f1'].tolist()
        num_enumerate*=len(np.unique(mean_f1_target))
        mean_f1.append(np.mean(mean_f1_target))

    print('number of possibilities ', num_enumerate)
    return np.mean(mean_f1)

print('normalized random mean ', get_mean(target_list,args.upperbound))
print('unnormalized random mean ', get_mean(target_list,args.upperbound,normalize=False))

def simulate(target_list,upperbound=args.upperbound,num_events=args.num_episodes,normalize=True):
    mean_f1_list=[]
    for i in range(num_events):
        if i%1000==0:
            print(i)
        mean_f1_episode=[]
        for target in target_list:
            target_df=pickle.load(open('../Pharmnn/data/dude/all/'+target+'/target_df.pkl','rb'))
            if normalize:
                target_df['f1']=target_df['f1']/target_df['f1'].max()
            f1_target=[]
            for num in range(3,upperbound):
                target_df_num=target_df[target_df['file'].str.startswith('pharmit_'+str(num)+'_')==1].sort_values(by='f1',ascending=False)
                #remove duplicate rows on file
                target_df_num=target_df_num.drop_duplicates(subset='file')
                f1_target+=target_df_num['f1'].tolist()
            mean_f1_episode.append(random.choice(f1_target))
        mean_f1_list.append(np.mean(mean_f1_episode))
    return mean_f1_list

def create_cumulative(mean_f1_list):
    mean_f1_list.sort()
    y_values=np.arange(len(mean_f1_list))/float(len(mean_f1_list))
    return mean_f1_list,y_values

mean_f1_list=simulate(target_list,num_events=args.num_episodes,normalize=True)
mean_f1_list,y_values=create_cumulative(mean_f1_list)

def p_value(mean_f1_list,y_values,f1_score=args.f1_score):
    for i in range(len(mean_f1_list)):
        if mean_f1_list[i]>f1_score:
            return y_values[i]
    return 0

print('normalized p-value ', p_value(mean_f1_list,y_values))


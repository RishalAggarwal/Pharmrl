#!/bin/bash

#job name
#SBATCH --job pharmrl_training
#SBATCH --partition dept_gpu
#SBATCH --gres=gpu:1
##SBATCH --array 1-30
##SBATCH --exclude g010,g011,g019,g013,g012
#SBATCH -w g016
#SBATCH --mail-user=ria43@pitt.edu
#SBATCH --mail-type=ALL
#SBATCH -c 4

# directory name where job will be run (on compute node)
#job_dir="${user}_${SLURM_JOB_ID}.dcb.private.net"

# creating directory on /scr folder of compute node
#mkdir /scr/$job_dir

# put date and time of starting job in a file
#date > date.txt

# put hostname of compute node in a file
#hostname > hostname.txt

# copy files on exit or interrupt
# make sure this is before your main program for it to always run on exit
#trap "echo 'copying files'; rsync -avz * ${SLURM_SUBMIT_DIR}" EXIT

# copy the submit file (and all other related files/directories)
#rsync -a ${SLURM_SUBMIT_DIR}/*.pkl /scr/${job_dir}


source activate phramnn
module load cuda/11.5
#python ./train_pharmnn.py --train_data data/chemsplit_train0.pkl --test_data data/chemsplit_test0.pkl  --wandb_name default_chemsplit0_large_256 --grid_dimension 15.5  --expand_width 0 --model models/default_chemsplit0_large_256_last_model.pkl --lr 0.00001
#python ./train_pharmnn.py --train_data data/chemsplit_train2_with_ligand.pkl --test_data data/chemsplit_test2_with_ligand.pkl  --wandb_name obabel_chemsplit2_2 --negative_data data/obabel_chemsplit_2_negatives_train.txt --batch_size 256 --model models/obabel_chemsplit2_last_model.pkl --lr 0.00001


#/net/pulsar/home/koes/rishal/.conda/envs/phramnn/bin/wandb agent --count 1 rishalaggarwal/pharmrl/22jqmayw
#python q-learning.py --wandb_run_name all_dude
#python q-learning.py --batch_norm=true --batch_size=45 --epsilon_decay=16000 --epsilon_min=0.017011998579683857 --epsilon_start=0.8733283346001617 --gamma=0.4650925534883208 --lr=0.000492841981715628 --max_radius=8 --memory_size=1014 --num_conv_layers=8 --num_episodes=30000 --pharm_pharm_radius=15 --protein_pharm_radius=12 --radius_embed_dim=97 --residual=true --reward_type=f1 --target_update=1 --tau=0.6232604485053497 --save_model True --train_file train_dude_obabel_1_2.txt --test_file test_dude_obabel_1_2.txt
python q-learning.py --batch_norm=true --batch_size=50 --epsilon_decay=5000 --epsilon_min=0.0175651687171413 --epsilon_start=0.5 --gamma=0.8636308152558942 --lr=0.00011835636800220085 --num_steps=5 --max_radius=9 --memory_size=1893 --num_conv_layers=6 --num_episodes=5000 --pharm_pharm_radius=12 --protein_pharm_radius=11 --radius_embed_dim=78 --residual=true --reward_type=f1 --target_update=2 --tau=0.6860764845515854 --return_reward dude_pharmit --top_dir /scr/rishal/dude/all --train_file data/dude_pharmrl_features_all_train.txt --test_file data/dude_pharmrl_features_all_test.txt --model_weights models/sweep_model.pt --save_model=true --wandb_run_name parallel_retrain_all_points --num_tests=20 --parallel=true --processes=4

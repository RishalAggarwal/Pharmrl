# Pharmrl  [![DOI](https://zenodo.org/badge/621970276.svg)](https://doi.org/10.5281/zenodo.14397445)

Deep Q-learning for forming an ideal pharmacophore from a given set of pharmacophore features using a SE(3) Equivariant Neural Network.


Google colab notebook: [https://colab.research.google.com/github/RishalAggarwal/PharmRL/blob/main/notebooks/pharmrl.ipynb](https://colab.research.google.com/github/RishalAggarwal/PharmRL/blob/main/notebooks/pharmrl.ipynb)


## Setting up datasets from SMILES strings

Lets say we have a set of directories in DUD-E with ```actives_final.ism``` and ```decoys_final.ism``` files. We need to generate conformers for these and then setup pharmit databases to train the models.

### Generating conformers

To generate conformers, use the ```rdconf.py``` script at ```https://github.com/dkoes/rdkit-scripts```.

Example usage: ```python rdconf.py decoys_final.ism decoys_final.sdf.gz --maxconfs 200```

To label the active conformers we can use the sdsorter binary at ```https://sourceforge.net/projects/sdsorter/```

Example usage: ```sdsorter -concatToTitle=_active actives_final.sdf.gz actives_labeled.sdf.gz```

### Pharmit database creation

The files need to be decompressed in each DUD-E directory

```gunzip decoys_final.sdf.gz```
```gunzip actives_labeled.sdf.gz```

Pharmit needs to be downloaded at  ```https://github.com/dkoes/pharmit/releases/download/v1.0/pharmit```

Run the following command to create a pharmit database for each directory

```pharmit dbcreate -in actives_final.sdf  -in decoys_final.sdf -reduceconfs=25 -dbdir pharmit_db```

A similar set of commands can be used to create pharmit databases for the other datasets used in the study.

## Setting up the environment

The environment yml file is at ```environment.yml``` but if it gets too tough to install, the environment can directly downloaded using:

```!gsutil cp gs://koes-data/pharmnnrl-env.tar```

## Training Pharmrl models

Before training the models, you will also have to untar the following files into the DUD-E directories. These contain the pharmit F1 scores of pharmacophores generated from enumeration of ligand features:

```tar -xzvf data/target_pharmit_scores.tar.gz```

To train the model on ligand only features, run the following command:

```python qlearning.py --batch_norm=true --batch_size=50 --epsilon_decay=1967 --epsilon_min=0.0175651687171413 --epsilon_start=0.8361945747695066 --gamma=0.8636308152558942 --lr=0.00011835636800220085 --max_radius=9 --memory_size=1893 --num_conv_layers=6 --num_episodes=3000 --pharm_pharm_radius=12 --protein_pharm_radius=11 --radius_embed_dim=78 --residual=true --reward_type=f1 --target_update=2 --tau=0.6860764845515854 --return_reward dataframe --top_dir /data/pharmnn_rl/dude/all --train_file data/dude_pharmrl_features_all_train.txt --test_file data/dude_pharmrl_features_all_test.txt --save_model=true --wandb_run_name dude_ligand_train --num_tests=20 --parallel=true --processes=12```

To fine tune the pretrained hyperparameter optimized model on CNN features, run the following command:

Example command: ```python qlearning.py --batch_norm=true --batch_size=50 --epsilon_decay=1967 --epsilon_min=0.0175651687171413 --epsilon_start=0.3 --gamma=0.8636308152558942 --lr=0.00011835636800220085 --max_radius=9 --memory_size=1893 --num_conv_layers=6 --num_episodes=3000 --pharm_pharm_radius=12 --protein_pharm_radius=11 --radius_embed_dim=78 --residual=true --reward_type=f1 --target_update=2 --tau=0.6860764845515854 --return_reward dude_pharmit --top_dir /data/dude/all --train_file data/dude_pharmrl_features_all_train.txt --test_file data/dude_pharmrl_features_all_test.txt --model_weights models/model_ligand.pt --save_model=true --wandb_run_name dude_train --num_tests=20 --parallel=true --processes=12```




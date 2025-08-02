import argparse
from pathlib import Path
import os
from time import time
import pandas as pd
import logging
import numpy as np
import matplotlib.pyplot as plt
import neps
from sklearn.metrics import accuracy_score

#from problem_project import ProjProblem
from automl.automl_cancer import AutoML
from automl.datasets_cancer import FashionDataset, FlowersDataset, EmotionsDataset, SkinCancerDataset
import optuna
from collections import defaultdict
from torch import nn
import sqlalchemy

logger = logging.getLogger(__name__)

def choose_network() -> nn.Module:
    print(f'\n{dataset_class._dataset_name}')
    automl = AutoML(seed=seed) #TODO fix seed thing
    
    hyperparams = defaultdict(int)
    networks = ['pretrained', 'scratch']
    best_acc = 0
    best = networks[0]

    for choice in networks:
        hyperparams['network'] = choice
        automl.fit(dataset_class, hyperparams, epochs=2, RESIZE_SIZE=resize)
        preds, labels = automl.predict(dataset_class)
        acc = accuracy_score(labels, preds)
        if acc > best_acc:
            best_acc = acc
            best = choice
        print(f"Accuracy of {choice} on test set: {acc}\n")
    print(f'BEST NETWORK: {best}\n\n')
    return best
		

def objective(trial: optuna.Trial):
    hyperparams = defaultdict(int)
	#TODO blur kernel size? or probability of blur? probability of rotation? learning rate schedule? (e.g. cut LR every # epochs... or other methods)
    # rot = trial.suggest_int('rot', 0, 60, step=15)
    # horflip = trial.suggest_float('horflip', 0, 0.4, step=0.1)
    # verflip = trial.suggest_float('verflip', 0, 0.4, step=0.1)
    # blur = trial.suggest_float('blur', low=0, high=3, step=1)
    # affine = trial.suggest_int('affine', 0, 60, step=15)
    #noise = trial.suggest_float('noise', low=0, high=0.2, step=0.05)
    # network = trial.suggest_categorical(name='network', choices=['scratch', 'pretrained'])
    # augments.update(dict(rot=rot, horflip=horflip))

    initial_lr = trial.suggest_float(name='initial_lr', low=1e-3, high=1e-2, log=True)
    # lr_decay = trial.suggest_float(name='lr_decay', low=0.1, high=0.5)
    # dropout = trial.suggest_float(name='dropout', low=0, high=0.4, step=0.1)
    # print(f'\n{dataset_class._dataset_name}')

    automl = AutoML(seed=seed) #TODO fix seed thing
    
    # hyperparams.update(dict(rot=rot, horflip=horflip, verflip=verflip, blur=blur, affine=affine))
    # hyperparams.update(dict(initial_lr=initial_lr, lr_decay=lr_decay, dropout=dropout, network=network))
    hyperparams.update(dict(initial_lr=initial_lr, network=network))
	#!
    automl.fit(dataset_class, hyperparams, epochs=epochs, RESIZE_SIZE=resize, trial=trial)
    preds, labels = automl.predict(dataset_class)
    if not np.isnan(labels).any(): #TODO remove
        acc = accuracy_score(labels, preds)
        # logger.info(f"Accuracy on test set: {acc}")
        print(f"Accuracy on test set: {acc}")
    else:
        print('DID NOT FIND LABELS')
    return acc

import logging, sys #!
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG, stream=sys.stdout) #! this turned out to do nothing
#logging.basicConfig(level=logging.INFO)


parser = argparse.ArgumentParser()
	
parser.add_argument(
	"--dataset",
	type=str,
	required=True,
	help="The name of the dataset to run on.",
	choices=["fashion", "flowers", "emotions", "cancer"]
)
	
parser.add_argument(
	"--trials",
	type=int,
	default=10,
	help="Number of NePS trials.",
)
	
parser.add_argument(
	"--epochs",
	type=int,
	default=5,
	help="Number of epochs per trial.",
)
	
parser.add_argument(
	"--resize",
	type=int,
	default=0,
	help="Resize image width/height to this value. If the images are smaller, this argument is ignored.",
)
	
parser.add_argument(
	"--seed",
	type=int,
	default=42,
	help="Resize image width/height to this value. If the images are smaller, this argument is ignored.",
)
	
parser.add_argument(
	"--replace",
	action='store_true',
	help="Resize image width/height to this value. If the images are smaller, this argument is ignored.",
)
	
parser.add_argument(
	"--delete",
	action='store_true',
	help="Resize image width/height to this value. If the images are smaller, this argument is ignored.",
)

args = parser.parse_args()


dataset = args.dataset
trials = args.trials #!#TODO TEMPORARY
epochs = args.epochs #!#TODO TEMPORARY
resize = args.resize #!#TODO TEMPORARY
replace = args.replace
delete = args.delete
seed = args.seed

match dataset:
	case "fashion":
		dataset_class = FashionDataset
	case "flowers":
		dataset_class = FlowersDataset
	case "emotions":
		dataset_class = EmotionsDataset
	case "cancer":
		dataset_class = SkinCancerDataset
	case _: #TODO allow new datasets
		raise ValueError(f"Invalid dataset: {dataset}")


size = min(dataset_class.width, resize) if resize != 0 else dataset_class.width
study_name = f'{dataset} ({size}x{size}), trials={trials}, epochs={epochs}'

res_dir = 'optuna_res'
output_file = f'{res_dir}/{study_name}.txt'
os.makedirs(res_dir,exist_ok=True)

assert not os.path.isfile(output_file), 'A .txt for this study already exists. Rename it.'

psql = "postgresql://diogwo:digypsql@localhost/automl_optuna"
if replace or delete:
	try:
		optuna.delete_study(study_name=study_name, storage=psql)
		print('previous study deleted\n')
		if delete:
			quit()
	except KeyError as err:
		pass
	except sqlalchemy.exc.OperationalError as err: #! you guys are not using psql. This makes it so you don't need to change anything
		pass
try:
	study = optuna.load_study(study_name=study_name, storage=psql)
except KeyError as err:
	# study = optuna.create_study(direction='maximize', study_name=study_name, storage=psql, 
	# 			sampler=optuna.samplers.TPESampler(multivariate=True, n_startup_trials=40))#,constant_liar=True)) #TODO sampler
	# study = optuna.create_study(direction='maximize', study_name=study_name, storage=psql, 
	# 			sampler=optuna.samplers.TPESampler(multivariate=True, n_ei_candidates=100,constant_liar=True),
	# 			pruner=optuna.pruners.HyperbandPruner())#,constant_liar=True)) #TODO sampler
	study = optuna.create_study(direction='maximize', study_name=study_name, storage=psql, 
				sampler=optuna.samplers.TPESampler(multivariate=True, n_ei_candidates=100,constant_liar=True)) #! constant_liar is for multiprocess
# study = optuna.create_study(direction='maximize', study_name=study_name) #TODO sampler
except sqlalchemy.exc.OperationalError as err:
	#! this is the only part of the try catch you are running. Ignore everything else (psql related)
	study = optuna.create_study(direction='maximize', study_name=study_name, 
				sampler=optuna.samplers.TPESampler(multivariate=True, n_ei_candidates=100,constant_liar=True),)

t_before_study = time()
network = choose_network() #!
study.optimize(func=objective, n_trials=trials)
# quit() #! REQUIRED FOR PARALLELIZING; RUN write_optuna.py AFTERWARDS 
#! (otherwise, will bug out due to incomplete trials: the ones the other process is still calculating!)
t_after_study = time()
timediff = t_after_study - t_before_study
duration_str = f'{timediff // 60:.0f}m{timediff % 60:.3f}s'

best = study.best_trial

output_str = f'{study_name}\n\n- study duration: {duration_str}\n\n'
output_str += f'-------- best --------\n\ntrial: {best.number}\n- {best.params}\naccuracy: {best.value}\n\n'

trials = study.get_trials()
try:
	sorted_trials = sorted(trials, key=lambda t: t.value, reverse=True)
except TypeError as err:
	print("Other processes are still running optuna. The last one will write the output file.")
	quit()
output_str += f'------ rankings ------\n\n'
for t in sorted_trials:
	params = {k: f'{v:.2f}' for k, v in t.params.items()}
	output_str += f'{t.number}: {t.value:.5f}; {params}\n'

output_str += '\n'

output_str += f'------- trials -------\n\n'

for t in trials:
	#print(f'trial {i}:\n- {t.params}\n- {t.value}\n')
	params = {k: f'{v:.2f}' for k, v in t.params.items()}
	output_str += f'trial {t.number}:\n- {params}\n- accuracy: {t.value:.5f}\n\n'

#print(f'best params: {best.params}\naccuracy: best.value')
#output_str += f'----- best -----\ntrial: {best.number}\n{best.params}\naccuracy: {best.value}\nstudy duration: {duration_str}'
print(output_str)


with open(output_file, mode = 'w') as f:
	f.write(output_str[:-1])

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

logger = logging.getLogger(__name__)

def objective(trial: optuna.Trial):
	#TODO blur kernel size? or probability of blur? probability of rotation? learning rate schedule? (e.g. cut LR every # epochs... or other methods)
    rot = trial.suggest_int('rot', 0, 60, step=15)
    horflip = trial.suggest_float('horflip', 0, 0.4, step=0.1)
    verflip = trial.suggest_float('verflip', 0, 0.4, step=0.1)
    blur = trial.suggest_float('blur', low=0, high=3, step=1)
    affine = trial.suggest_int('affine', 0, 60, step=15)
    #noise = trial.suggest_float('noise', low=0, high=0.2, step=0.05)
    print(f'\n{dataset_class._dataset_name}')
    automl = AutoML(seed=seed) #TODO fix seed thing
    
    #augments = dict(rot=rot, horflip=horflip, blur=blur, noise=noise)
    #augments = dict(rot=rot, horflip=horflip, verflip=verflip, noise=noise)
    augments = defaultdict(int) #TODO default dict to make removing stuff easier (will need to cleanup automl2 later anyways, though.)
    augments.update(dict(rot=rot, horflip=horflip, verflip=verflip, blur=blur, affine=affine))
    # augments.update(dict(rot=rot, horflip=horflip))
	#!
    automl.fit(dataset_class, augments, epochs=epochs, RESIZE_SIZE=resize)
    preds, labels = automl.predict(dataset_class)
    if not np.isnan(labels).any(): #TODO remove
        acc = accuracy_score(labels, preds)
        # logger.info(f"Accuracy on test set: {acc}")
        print(f"Accuracy on test set: {acc}")
    else:
        print('DID NOT FIND LABELS')
    return acc


# pipeline_space = dict(
# 	rot = neps.Integer(
# 		lower=0,
# 		upper=90,
# 		prior=30,
#         prior_confidence='medium',
# 	),
# 	# horflip = neps.Float(
# 	# 	lower=0,
# 	# 	upper=0.4,
# 	# 	prior=0.2,
# 	# ),
# 	horflip = neps.Categorical(
# 		choices=[0.1, 0.2, 0.3, 0.4],
# 		prior=0.2,
# 	),
#     blur = neps.Categorical( #TODO allow bigger sigmas
#         # choices=[0, 0.1, 0.2],
#         choices=[0, 0.2, 0.5, 1],
# 	), #TODO float makes it die? try giving it a prior, mby it makes it work (although the bug is still there obv. and probably still is w/ Categorical)
#     noise = neps.Categorical( #TODO allow bigger sigmas
#         #choices=[0, 0.1, 0.2],
#         choices=[0, 0.2, 0.5, 1],
# 	),
#     # blur = neps.Float(
#     #     lower=0,
#     #     upper=0.2,
# 	# )#TODO add tanh, etc.
# 	#dataset_class=datasets.EmotionsDataset, #!
# )
import logging, sys #!
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG, stream=sys.stdout) #!
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

output_file = f'optuna_res/{study_name}.txt'

assert not os.path.isfile(output_file), 'A .txt for this study already exists. Rename it.'

import torch
torch.multiprocessing.set_start_method('spawn')
psql = "postgresql://diogo:digypsql@localhost/automl_optuna"
if replace or delete:
	try:
		optuna.delete_study(study_name=study_name, storage=psql)
		print('previous study deleted\n')
		if delete:
			quit()
	except KeyError as err:
		pass
try:
	study = optuna.load_study(study_name=study_name, storage=psql)
except KeyError as err:
	# study = optuna.create_study(direction='maximize', study_name=study_name, storage=psql, 
	# 			sampler=optuna.samplers.TPESampler(multivariate=True, n_startup_trials=40))#,constant_liar=True)) #TODO sampler
	study = optuna.create_study(direction='maximize', study_name=study_name, storage=psql, 
				sampler=optuna.samplers.TPESampler(multivariate=True, n_startup_trials=20, n_ei_candidates=200,constant_liar=True))#,constant_liar=True)) #TODO sampler
# study = optuna.create_study(direction='maximize', study_name=study_name) #TODO sampler

t_before_study = time()
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

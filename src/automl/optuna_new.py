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
from automl2 import AutoML
from datasets import FashionDataset, FlowersDataset, EmotionsDataset
import optuna

logger = logging.getLogger(__name__)

def objective(trial: optuna.Trial):
    rot = trial.suggest_int('rot', 0, 90, step=5)
    horflip = trial.suggest_float('horflip', 0, 0.4, step=0.05)
    blur = trial.suggest_float('blur', low=0, high=2, step=0.5)
    noise = trial.suggest_float('noise', low=0, high=0.2, step=0.05)
    print(f'\n{dataset_class._dataset_name}')
    automl = AutoML(42) #TODO fix seed thing
    
    augments = dict(rot=rot, horflip=horflip, blur=blur, noise=noise)
	#!
    automl.fit(dataset_class, augments, epochs=epochs, RESIZE_SIZE=resize)
    preds, labels = automl.predict(dataset_class)
    if not np.isnan(labels).any(): #TODO remove
        acc = accuracy_score(labels, preds)
        # logger.info(f"Accuracy on test set: {acc}")
        print(f"Accuracy on test set: {acc}")
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
	choices=["fashion", "flowers", "emotions"]
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

args = parser.parse_args()


dataset = args.dataset
trials = args.trials #!#TODO TEMPORARY
epochs = args.epochs #!#TODO TEMPORARY
resize = args.resize

match dataset:
	case "fashion":
		dataset_class = FashionDataset
	case "flowers":
		dataset_class = FlowersDataset
	case "emotions":
		dataset_class = EmotionsDataset
	case _: #TODO allow new datasets
		raise ValueError(f"Invalid dataset: {dataset}")


size = min(dataset_class.width, resize)
study_name = f'{dataset} ({size}x{size}), trials={trials}, epochs={epochs}'
study = optuna.create_study(direction='maximize', study_name=study_name)

t_before_study = time()
study.optimize(func=objective, n_trials=trials)
t_after_study = time()
timediff = t_after_study - t_before_study
duration_str = f'{timediff // 60:.0f}m{timediff % 60:.3f}s'

best = study.best_trial

output_str = f'{study_name}\n\n- study duration: {duration_str}\n\n'
output_str += f'-------- best --------\n\n\ntrial: {best.number}\n- {best.params}\naccuracy: {best.value}\n\n\n'
output_str += f'------- trials -------\n\n\n'

trials = study.get_trials()
for i in range(len(trials)):
	t = trials[i]
	#print(f'trial {i}:\n- {t.params}\n- {t.value}\n')
	output_str += f'trial {i}:\n- {t.params}\n- accuracy: {t.value}\n\n'

#print(f'best params: {best.params}\naccuracy: best.value')
#output_str += f'----- best -----\ntrial: {best.number}\n{best.params}\naccuracy: {best.value}\nstudy duration: {duration_str}'
print(output_str)


with open(f'optuna_res/{study_name}.txt', mode = 'w') as f:
    f.write(output_str[:-1])

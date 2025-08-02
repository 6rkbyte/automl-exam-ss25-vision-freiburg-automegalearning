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
import re

logger = logging.getLogger(__name__)
logging.getLogger("PIL").setLevel(logging.WARNING)

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
	"--index",
	type=int,
	default=1,
	help="Index of the study file within those in this particular 'config' (dataset, trials, epochs)",
)
	
parser.add_argument(
	"--processes",
	type=int,
	default=1,
	help="Number of processes in this study (in case I've renamed the file accordingly)",
)
	
parser.add_argument(
	"--write",
	action='store_true',
	help="Write .txt? (use if the optuna script failed to do so)",
)
	
parser.add_argument(
	"--nohtml",
	action='store_true',
	help="Do not output html files",
)

args = parser.parse_args()


dataset = args.dataset
trials = args.trials #!#TODO TEMPORARY
epochs = args.epochs #!#TODO TEMPORARY
resize = args.resize #!#TODO TEMPORARY
index = args.index
processes = args.processes
write = args.write
nohtml = args.nohtml

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

if processes == 1:
	file_path = f'optuna_res/{study_name}_{index}.txt'
else:
	file_path = f'optuna_res/{study_name}, processes=2_{index}.txt'

#trial_acc_pairs = []
trials = []
accs = []
try:
	with open(file_path, mode = 'r') as f:
		lines = f.readlines()
		# current_trial = 0
		# current_acc = np.nan
		for line in lines:
			trial_match = re.search(r'trial (\d+):', line)
			accuracy_match = re.search(r'- accuracy: (\d+.\d+)', line)
			if trial_match:
				current_trial = int(trial_match.group(1))
			if accuracy_match:
				current_trial_acc = float(accuracy_match.group(1))
				#trial_acc_pairs.append([current_trial, current_trial_acc])
				trials.append(current_trial) # must be here in case a None is encountered
				accs.append(current_trial_acc)

		import matplotlib as mpl
		mpl.set_loglevel("warning")
		# trials = trial_acc_pairs.keys()
		# accs = trial_acc_pairs.values()
		plt.plot(trials, accs)
		plt.ylim(bottom=0)
		plt.show()

except FileNotFoundError:
	print(f'file {file_path} not found. Skipping the accuracy per trial graph.')
		
# for p in trial_acc_pairs:
# 	print(p)


psql = "postgresql://diogo:digypsql@localhost/automl_optuna"
print(f'loading study: {study_name}\n')
study = optuna.load_study(study_name=study_name, storage=psql)
optim_history = optuna.visualization.plot_optimization_history(study)
param_importances = optuna.visualization.plot_param_importances(study)

import plotly.io as pio
# print(pio.renderers)
pio.renderers.default = "browser"
# print(optuna.get_all_study_names(storage=psql))
optim_history.show()
param_importances.show()
if not nohtml:
	optim_history.write_html(f"optuna_res/plots/{study_name} - optimization history.html")
	param_importances.write_html(f"optuna_res/plots/{study_name} - param importances.html")



if not write:
	quit()

output_file = f'optuna_res/{study_name}.txt'
assert not os.path.isfile(output_file), 'A .txt for this study already exists. Rename it.'

trials = study.get_trials()
best = study.best_trial
time_before_study = trials[0].datetime_start.timestamp()
time_after_study = trials[-1].datetime_complete.timestamp()
timediff = time_after_study - time_before_study
duration_str = f'{timediff // 60:.0f}m{timediff % 60:.3f}s'

output_str = f'{study_name}\n\n- study duration: {duration_str}\n\n'
output_str += f'-------- best --------\n\ntrial: {best.number}\n- {best.params}\naccuracy: {best.value}\n\n'

try:
	sorted_trials = sorted(trials, key=lambda t: t.value, reverse=True)
except TypeError as err:
	print("Trial sorting failed. There likely is a None in there.")
	good_trials = [t for t in trials if t.value is not None]
	bad_trials = [t for t in trials if t.value is None]
	sorted_trials = sorted(good_trials, key=lambda t:t.value, reverse=True)
	sorted_trials += bad_trials
	# quit()
# sorted_trials = trials
output_str += f'------ rankings ------\n\n'
for t in sorted_trials:
	number = t.number if t.number is not None else 'None'
	value = f'{t.value:.5f}' if t.value is not None else 'None'
	params = {k: f'{v:.2f}' for k, v in t.params.items()} if t.params is not None else 'None'
	output_str += f'{t.number}: {value}; {params}\n'

output_str += '\n'

output_str += f'------- trials -------\n\n'

for t in trials:
	#print(f'trial {i}:\n- {t.params}\n- {t.value}\n')
	number = t.number if t.number is not None else 'None'
	value = f'{t.value:.5f}' if t.value is not None else 'None'
	params = {k: f'{v:.2f}' for k, v in t.params.items()} if t.params is not None else 'None'
	output_str += f'trial {number}:\n- {params}\n- accuracy: {value}\n\n'

#print(f'best params: {best.params}\naccuracy: best.value')
#output_str += f'----- best -----\ntrial: {best.number}\n{best.params}\naccuracy: {best.value}\nstudy duration: {duration_str}'
print(output_str)

with open(output_file, mode = 'w') as f:
	f.write(output_str[:-1])

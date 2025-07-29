import argparse
from pathlib import Path
import os
import time
import pandas as pd
import logging
import numpy as np
import matplotlib.pyplot as plt
import neps
from sklearn.metrics import accuracy_score

#from problem_project import ProjProblem
from automl2 import AutoML
import datasets

logger = logging.getLogger(__name__)

def evaluate_pipeline(**config):
    rot = config.pop('rot')
    horflip = config.pop('horflip')
    blur = config.pop('blur')
    #dataset_class = config.pop('dataset_class')
    # dataset_class = datasets.EmotionsDataset
    # dataset_class = datasets.FlowersDataset
    dataset_class = datasets.FashionDataset
    print(f'\n{dataset_class._dataset_name}')
    automl = AutoML(42) #TODO fix seed thing
    
    augments = dict(rot=rot, horflip=horflip, blur=blur)
    # augments = dict(rot=rot, horflip=horflip)
    automl.fit(dataset_class, augments)
    preds, labels = automl.predict(dataset_class)
    if not np.isnan(labels).any():
        acc = accuracy_score(labels, preds)
        logger.info(f"Accuracy on test set: {acc}")

    return {'loss': -acc, 'objective_to_minimize': -acc}


def run_neps(evaluate_pipeline, pipeline_space, root_dir='./temp_neps'):

	neps.run(
		evaluate_pipeline=evaluate_pipeline,
		pipeline_space = pipeline_space,
		root_directory=root_dir,
		max_evaluations_total=20,
        #overwrite_working_directory=True,
	)


pipeline_space = dict(
	rot = neps.Integer(
		lower=0,
		upper=90,
		prior=30,
        prior_confidence='medium',
	),
	horflip = neps.Float(
		lower=0,
		upper=0.4,
		prior=0.2,
	),
    blur = neps.Categorical(
        choices=[0, 0.1, 0.2],
	) #TODO float makes it die? try giving it a prior, mby it makes it work (although the bug is still there obv. and probably still is w/ Categorical)
    # blur = neps.Float(
    #     lower=0,
    #     upper=0.2,
	# )#TODO add tanh, etc.
	#dataset_class=datasets.EmotionsDataset, #!
)
import logging, sys #!
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG, stream=sys.stdout)
#logging.basicConfig(level=logging.INFO)

run_neps(evaluate_pipeline=evaluate_pipeline, pipeline_space=pipeline_space)

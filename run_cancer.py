"""An example run file which trains a dummy AutoML system on the training split of a dataset
and logs the accuracy score on the test set.

In the example data you are given access to the labels of the test split, however
in the test dataset we will provide later, you will not have access
to this and you will need to output your predictions for the images of the test set
to a file, which we will grade using github classrooms!
"""
from __future__ import annotations

from pathlib import Path
from sklearn.metrics import accuracy_score
import numpy as np
# from automl.automl2 import AutoML
#TODO cancer
from automl.automl_cancer import AutoML
from automl.datasets_cancer import FashionDataset, FlowersDataset, EmotionsDataset, SkinCancerDataset
#TODO kornia
# from automl.automl_kornia import AutoML
# from automl.datasets_kornia import FashionDataset, FlowersDataset, EmotionsDataset
import argparse
from collections import defaultdict
import logging, sys

logger = logging.getLogger(__name__)


def main(
    dataset: str,
    output_path: Path,
    seed: int,
):
    match dataset:
        case "fashion":
            dataset_class = FashionDataset
        case "flowers":
            dataset_class = FlowersDataset
        case "emotions":
            dataset_class = EmotionsDataset
        case "cancer":
            dataset_class = SkinCancerDataset
        case _:
            raise ValueError(f"Invalid dataset: {args.dataset}")

    logger.info("Fitting AutoML")

    # You do not need to follow this setup or API it's merely here to provide
    # an example of how your automl system could be used.
    # As a general rule of thumb, you should **never** pass in any
    # test data to your AutoML solution other than to generate predictions.
    automl = AutoML(seed=seed)
    # load the dataset and create a loader then pass it
    automl.fit(dataset_class, augments, epochs=epochs, RESIZE_SIZE=resize)
    # Do the same for the test dataset
    test_preds, test_labels = automl.predict(dataset_class)

    # Write the predictions of X_test to disk
    # This will be used by github classrooms to get a performance
    # on the test set.
    logger.info("Writing predictions to disk")
    with output_path.open("wb") as f:
        np.save(f, test_preds)

    # check if test_labels has missing data


    if not np.isnan(test_labels).any():
        acc = accuracy_score(test_labels, test_preds)
        logger.info(f"Accuracy on test set: {acc}")
    else:
        # This is the setting for the exam dataset, you will not have access to the labels
        logger.info(f"No test split for dataset '{dataset}'")


parser = argparse.ArgumentParser()

parser.add_argument(
	"--dataset",
	type=str,
	required=True,
	help="The name of the dataset to run on.",
	choices=["fashion", "flowers", "emotions", "cancer"]
)
parser.add_argument(
	"--output-path",
	type=Path,
	default=Path("predictions.npy"),
	help=(
		"The path to save the predictions to."
		" By default this will just save to './predictions.npy'."
	)
)
parser.add_argument(
	"--seed",
	type=int,
	default=42,
	help=(
		"Random seed for reproducibility if you are using and randomness,"
		" i.e. torch, numpy, pandas, sklearn, etc."
	)
)
parser.add_argument(
	"--quiet",
	action="store_true",
	help="Whether to log only warnings and errors."
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

epochs = args.epochs
resize = args.resize
augments = defaultdict(int) #TODO default dict to make removing stuff easier (will need to cleanup automl2 later anyways, though.)
#! NOTE: DataLoader num_workers affects accuracy (randomness in the application of transforms)

# augments.update({'rot': 5, 'horflip': 0.25, 'verflip': 0, 'blur': 0}) #! -emotions (the blur makes it significantly worse, btw)
# augments.update({'rot': 5.00, 'horflip': 0.00, 'verflip': 0.00, 'blur': 0.50}) #! +emotions {'rot': '5.00', 'horflip': '0.00', 'verflip': '0.00', 'blur': '0.50'} (here, the blur makes it better)
# augments.update({'horflip': 0.20}) #! +emotions {'rot': '5.00', 'horflip': '0.00', 'verflip': '0.00', 'blur': '0.50'} (here, the blur makes it better)
# augments.update({'rot': 20, 'horflip': 0.4, 'verflip': 0, 'blur': 0.5})
# augments.update({'rot': 20, 'horflip': 0.4, 'verflip': 0, 'blur': 0}) #! +flowers
# augments.update({'rot': 20, 'horflip': 0.4, 'verflip': 0, 'blur': 0, 'affine': 40}) #! ++flowers (manual, not optuna)
# augments.update({'rot': 0, 'horflip': 0.3, 'verflip': 0, 'blur': 0, 'affine': 15}) #! -flowers (baited, looked better in optuna results)
# augments.update({'rot': 30, 'horflip': 0.2, 'verflip': 0.05, 'blur': 0.5}) #! -flowers {'rot': 30, 'horflip': 0.2, 'verflip': 0.05, 'blur': 0.5} (looks like blur makes it worse, though.)
# augments.update({'rot': 20, 'horflip': 0.4, 'verflip': 0, 'elastic': 0.1})
# augments.update({'rot': 20, 'horflip': 0.4, 'verflip': 0, 'gray': 0.03})
# augments.update({'rot': 0, 'horflip': 0, 'verflip': 0, 'gray': 0.03})
# augments.update({'horflip': 0.1, 'verflip': 0.1})
#augments.update({'rot': 20, 'horflip': 0.4}) #! cancer
# augments.update({'rot': 5, 'horflip': 0.4, 'verflip': 0.1, 'blur': 2.0}) #! cancer (try also kernel size 5x5)
augments.update({'rot': 5, 'horflip': 0.4, 'verflip': 0.1, 'blur': 3.0}) #! cancer (try also kernel size 5x5)
# augments.update({'blur': 2.0}) #! cancer (try also kernel size 5x5)
#{'rot': 5, 'horflip': 0.4, 'verflip': 0.1, 'blur': 2.0}
# augments.update({'rot': 20, 'horflip': 0.4, 'elastic':0.1})
# augments.update({'horflip': 0.20, 'elastic':0.1})
# augments.update({'affine': 10})

if not args.quiet:
	logging.basicConfig(level=logging.DEBUG, stream=sys.stdout)
else:
	logging.basicConfig(level=logging.WARNING)

logger.info(
	f"Running dataset {args.dataset}"
	f"\n{args}"
)

main(
	dataset=args.dataset,
	output_path=args.output_path,
	seed=args.seed,
)
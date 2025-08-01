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
from automl.automl import AutoML
import argparse
from torch.utils.data import DataLoader
from torchvision import transforms
import torch

import logging

from automl.datasets import FashionDataset, FlowersDataset, EmotionsDataset

logger = logging.getLogger(__name__)


def main(
    dataset: str,
    output_path: Path,
    seed: int,
    path: Path,
    networkType="CNN",
    epochs=100,
):
    match dataset:
        case "fashion":
            dataset_class = FashionDataset
        case "flowers":
            dataset_class = FlowersDataset
        case "emotions":
            dataset_class = EmotionsDataset
        case _:
            raise ValueError(f"Invalid dataset: {args.dataset}")

    #logger.info("Fitting AutoML")

    # You do not need to follow this setup or API it's merely here to provide
    # an example of how your automl system could be used.
    # As a general rule of thumb, you should **never** pass in any
    # test data to your AutoML solution other than to generate predictions.
    # Initialize AutoML
    automl = AutoML(seed=seed, networkType=networkType)

    # Reuse same transforms
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
    ])
    # Dataset
    train_dataset = dataset_class(root="./data", split='train', download=True, transform=transform)
    val_dataset = dataset_class(root="./data", split='test', download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    # ✅ 1. Tune the model first
    best_params = automl.tune(dataset_class=dataset_class,train_loader=train_loader,val_loader =val_loader, n_trials=100)
    print("Best hyperparameters found:", best_params)

    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        type=str,
        default="flowers",
        help="The name of the dataset to run on.",
        choices=["fashion", "flowers", "emotions"]
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
        "--path",
        type=Path,
        default=Path("training_log_CNN_flower.csv"),
        help=(
        )
    )
    parser.add_argument(
        "--networkType",
        type=str,
        default="CNN",
        help=(
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
        "--epochs",
        type=int,
        default=100,
        help=(
        )
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Whether to log only warnings and errors."
    )

    args = parser.parse_args()

    #if not args.quiet:
    #    logging.basicConfig(level=logging.INFO)
    #else:
    #    logging.basicConfig(level=logging.WARNING)

    #logger.info(
    #    f"Running dataset {args.dataset}"
    #    f"\n{args}"
    #)  
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    full_path = results_dir / args.path
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(torch.__version__)
    print(torch.cuda.is_available())
    print(torch.cuda.device_count())
    print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU")
    main(
        dataset=args.dataset,
        output_path=args.output_path,
        networkType=args.networkType,
        epochs=args.epochs,
        path=full_path,
        seed=args.seed,
    )
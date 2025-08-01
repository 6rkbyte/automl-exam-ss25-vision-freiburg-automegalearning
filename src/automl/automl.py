"""AutoML class for regression tasks.

This module contains an example AutoML class that simply returns predictions of a quickly trained MLP.
You do not need to use this setup, and you can modify this however you like.
"""
from __future__ import annotations

from typing import Any, Tuple

import torch
import random
import numpy as np
import logging

from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms

from automl.dummy_model import DummyNN
from automl.imitation_model import ResidualConvBlock, CNN
from automl.models import CNN_flex
from automl.utils import calculate_mean_std, init_log_file, log_epoch_results
from sklearn.metrics import accuracy_score
from pathlib import Path
import optuna
from optuna.pruners import HyperbandPruner


#logger = logging.getLogger(__name__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class AutoML:

    def __init__(
        self,
        seed: int,
        networkType: str,
    ) -> None:
        self.seed = seed
        self.networkType = networkType
        self._model: nn.Module | None = None
        self.best_params_: dict = {}
    def tune(self, dataset_class, train_loader, val_loader, n_trials=30, timeout=None):
        print(f"Using device: {device}")
        def objective(trial):
            # Sample hyperparameters
            lr = trial.suggest_float("lr", 5e-5, 1e-3, log=True)
            linear_n_layers = trial.suggest_int("linear_n_layers", 1, 2)
            conv_n_layers = trial.suggest_int("conv_n_layers", 1, 3)
            #optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "RMSprop", "SGD"])

            params = {
                "lr": lr,
                "linear_n_layers": linear_n_layers,
                "conv_n_layers": conv_n_layers,
            }
            for i in range(linear_n_layers):
                params[f"n_units_l{i}"] = trial.suggest_int(f"n_units_l{i}", 32, 600)
                params[f"dropout_l{i}"] = trial.suggest_float(f"dropout_l{i}", 0.0, 0.4)

            model = CNN_flex(in_channels=3, num_classes=dataset_class.num_classes, params=params).to(device)
            #optimizer_class = getattr(torch.optim, optimizer_name)
            #optimizer = optimizer_class(model.parameters(), lr=lr)
            optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999))
            criterion = nn.CrossEntropyLoss()

            for epoch in range(5):  # Fast tuning
                model.train()
                for data, target in train_loader:
                    data, target = data.to(device), target.to(device)
                    optimizer.zero_grad()
                    output = model(data)
                    loss = criterion(output, target)
                    loss.backward()
                    optimizer.step()

                # Validation for pruning
                model.eval()
                preds, targets = [], []
                with torch.no_grad():
                    for data, target in val_loader:
                        data, target = data.to(device), target.to(device)
                        output = model(data)
                        pred = output.argmax(1)
                        preds.append(pred.cpu().numpy())
                        targets.append(target.cpu().numpy())

                preds = np.concatenate(preds)
                targets = np.concatenate(targets)
                val_acc = accuracy_score(targets, preds)

                # Report to Optuna + pruning
                trial.report(val_acc, epoch)
                if trial.should_prune():
                    raise optuna.exceptions.TrialPruned()

            return val_acc

        # Use a custom pruner (e.g., MedianPruner)
        pruner = HyperbandPruner()
        study = optuna.create_study(direction="maximize", pruner=pruner)
        study.optimize(objective, n_trials=n_trials, timeout=timeout)

        self.best_params_ = study.best_trial.params
        return self.best_params_
    
    def fit(
        self,
        dataset_class: Any,
        epochs: int,
        path: Path,
    ) -> AutoML:
        """A reference/toy implementation of a fitting function for the AutoML class.
        """
        # set seed for pytorch training
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)

        # Ensure deterministic behavior in CuDNN
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        self._transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(*calculate_mean_std(dataset_class)),
            ]
        )
        #training log
        log_path = path

        init_log_file(log_path, headers=["Train_Acc","Val_Acc"])

        train_dataset = dataset_class(
            root="./data",
            split='train',
            download=True,
            transform=self._transform
        )

        val_dataset = dataset_class(
            root="./data",
            split='test',
            download=True,
            transform=self._transform
        )

        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=100, shuffle=False)

        input_size = dataset_class.width * dataset_class.height * dataset_class.channels

        # Choose model and hyperparams

        if self.networkType == "CNN":
            model = CNN(history_length=dataset_class.channels, n_classes=dataset_class.num_classes)
        elif self.networkType == "Dummy":
            model = DummyNN(input_size, dataset_class.num_classes)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.0001)
        
        
        for epoch in range(epochs):
            loss_per_batch = []
            all_preds = []
            all_targets = []

            #training
            model.train()
            for _, (data, target) in enumerate(train_loader):
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                loss_per_batch.append(loss.item())

                _, predicted = torch.max(output, 1)
                all_preds.extend(predicted.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
            train_acc = accuracy_score(all_targets, all_preds)

            #temporary validation
            
            predictions = []
            labels = []
            model.eval()
            with torch.no_grad():
                for data, target in val_loader:
                    output = model(data)
                    predicted = torch.argmax(output, 1)
                    labels.append(target.numpy())
                    predictions.append(predicted.numpy())
            predictions = np.concatenate(predictions)
            labels = np.concatenate(labels)
            test_acc = accuracy_score(labels, predictions)
            #logger.info(f"Epoch {epoch + 1}, Accuracy: {accuracy}, Loss: {np.mean(loss_per_batch)}")
            print(f"Epoch {epoch + 1}, Loss: {np.mean(loss_per_batch)}, Train_accuracy: {train_acc:.3f}, Val_accuracy: {test_acc:.3f},")
            log_epoch_results(log_path, [train_acc, test_acc])
        model.eval()
        self._model = model

        return self

    def predict(self, dataset_class) -> Tuple[np.ndarray, np.ndarray]:
        """A reference/toy implementation of a prediction function for the AutoML class.
        """
        dataset = dataset_class(
            root="./data",
            split='test',
            download=True,
            transform=self._transform
        )
        data_loader = DataLoader(dataset, batch_size=100, shuffle=False)
        predictions = []
        labels = []
        self._model.eval()
        with torch.no_grad():
            for data, target in data_loader:
                output = self._model(data)
                predicted = torch.argmax(output, 1)
                labels.append(target.numpy())
                predictions.append(predicted.numpy())
        predictions = np.concatenate(predictions)
        labels = np.concatenate(labels)
        
        return predictions, labels

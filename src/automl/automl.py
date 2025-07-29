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

from automl.dummy_model import DummyNN, CNN
from automl.utils import calculate_mean_std


logger = logging.getLogger(__name__)


class AutoML:

    def __init__(
        self,
        seed: int,
    ) -> None:
        self.seed = seed
        self._model: nn.Module | None = None

    def fit(
        self,
        dataset_class: Any,
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
        dataset = dataset_class(
            root="./data",
            split='train',
            download=True,
            transform=self._transform
        )
        train_loader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=4, pin_memory=True)

        input_size = dataset_class.width * dataset_class.height * dataset_class.channels

        #model = DummyNN(input_size, dataset_class.num_classes)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = CNN(dataset_class.channels, dataset_class.num_classes).to(device)
        print(next(model.parameters()).device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.003)
        
        model.train()
        predictions = [] #!
        labels = [] #!
        from time import time
        t1 = time()
        for epoch in range(5):
            t11 = time()
            loss_per_batch = []
            for _, (data, target) in enumerate(train_loader):
                data = data.to(device)
                target = target.to(device)
                
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                loss_per_batch.append(loss.item())

                predicted = torch.argmax(output, 1)   #! accuracy
                predicted = predicted.to('cpu')
                target = target.to('cpu')
                
                predictions.append(predicted.numpy()) #! accuracy
                labels.append(target.numpy())         #! accuracy
            predictions = np.concatenate(predictions)
            labels = np.concatenate(labels)
            
            t12 = time()
            logger.info(f"Epoch {epoch + 1}, Loss: {np.mean(loss_per_batch)}, Time: {t12 - t11}")
			#!  accuracy & time
            from sklearn.metrics import accuracy_score
            if not np.isnan(labels).any():
                acc = accuracy_score(labels, predictions)
                logger.info(f"Accuracy on train set: {acc}\n")
                labels, predictions = [], []
            #!/ accuracy & time
        model.eval()
        self._model = model
        t2 = time()
        print(f'took {t2-t1} seconds')

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
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        with torch.no_grad():
            for data, target in data_loader:
                output = self._model(data)
                predicted = torch.argmax(output, 1)
                
                predicted = predicted.to('cpu')
                target = target.to('cpu')
                
                labels.append(target.numpy())
                predictions.append(predicted.numpy())
        predictions = np.concatenate(predictions)
        labels = np.concatenate(labels)
        
        return predictions, labels

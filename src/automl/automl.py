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
from automl.imitation_model import CNN
from automl.pretrained import Mobilenet
from automl.vit import Vit
from automl.utils import calculate_mean_std
from automl.utils import TrainLog


logger = logging.getLogger(__name__)
device = "cuda:4" if torch.cuda.is_available() else "cpu"
#device = "cpu"


class AutoML:

    def __init__(
        self,
        seed: int,
    ) -> None:
        self.seed = seed
        self._model: nn.Module | None = None
        self.train_log: TrainLog = TrainLog()

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
        train_loader = DataLoader(dataset, batch_size=512, shuffle=True) # 64

        input_size = dataset_class.width * dataset_class.height * dataset_class.channels

        #model = DummyNN(input_size, dataset_class.num_classes)
        #model = CNN(dataset_class.channels, dataset_class.num_classes)
        model = Mobilenet(dataset_class.channels, dataset_class.num_classes)
        #model = Vit(dataset_class.channels, dataset_class.num_classes)
        
        model = model.to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.003) # 0.003
        
        model.train()
        for epoch in range(20):
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
            loss_mean = np.mean(loss_per_batch)
            logger.info(f"Epoch {epoch + 1}, Loss: {loss_mean}")
            self.train_log.append("train", epoch, loss_mean)
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
        data_loader = DataLoader(dataset, batch_size=100, shuffle=False) # 100
        predictions = []
        labels = []
        self._model.eval()
        with torch.no_grad():
            for data, target in data_loader:
                data = data.to(device)
                target = target.to(device)
                
                output = self._model(data)
                predicted = torch.argmax(output, 1)
                labels.append(target.cpu().numpy())
                predictions.append(predicted.cpu().numpy())
        predictions = np.concatenate(predictions)
        labels = np.concatenate(labels)
        
        return predictions, labels

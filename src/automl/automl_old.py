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

from automl import imitation_model
from torchvision.models import shufflenet_v2_x1_0, ShuffleNet_V2_X1_0_Weights
import torchvision.transforms as transforms

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

        # self._transform = transforms.Compose(
        #     [
        #         transforms.ToTensor(),
        #         transforms.Normalize(*calculate_mean_std(dataset_class)),
        #     ]
        # )
        # transform = transforms.Compose([
        #     transforms.Resize((res, res)),
        #     #transforms.Grayscale(num_output_channels=3),
        #     #! transforms.Lambda(lambda x: x.convert("RGB")), #! convert to 3 channels for resnet (PIL Image::convert())
        #     #TODO try out like this
		# 	#TODO also, mby do "if not grayscale, ColorJitter(hue...)"
        #     #TODO OR simply put this in the search space... hue = [0, 0.2]
        #     #transforms.ColorJitter(brightness=0.2, saturation=0.2, hue=0.2),
        #     #transforms.ColorJitter(hue=0.2),
        #     transforms.RandomRotation(degrees=30, interpolation=transforms.InterpolationMode.BICUBIC),
        #     transforms.RandomHorizontalFlip(p=0.2),
        #     transforms.ToTensor(),
        #     transforms.Normalize(*calculate_mean_std(dataset_class)),
        # ])
        # full_res_transform = transforms.Compose([ #TODO is this useful?
        #     transforms.RandomRotation(degrees=30, interpolation=transforms.InterpolationMode.BICUBIC),
        #     transforms.ToTensor(),
        #     transforms.Normalize(*calculate_mean_std(dataset_class)),
        # ])
        transform_list = [
            transforms.RandomRotation(degrees=41, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(p=0.316),
            transforms.ToTensor(),
            transforms.Normalize(*calculate_mean_std(dataset_class)),
		]
        #!  size
        res = 0 # 0 (fullres), 84, 112, 224
        res = (res, res) if isinstance(res, int) else res
        fullres = (res[0] == 0)
        if not fullres:
            transform_list = [transforms.Resize(res)] + transform_list
        #self._transform = full_res_transform if fullres else transform
        self._transform = transforms.Compose(transform_list)
		#!/ size
        epochs = 20 #!
        dataset = dataset_class(
            root="./data",
            split='train',
            download=True,
            transform=self._transform
        )
        train_loader = DataLoader(dataset, batch_size=64, shuffle=True, pin_memory=True, num_workers=4)
        size = (dataset_class.width, dataset_class.height) if fullres else res #! size
        logger.info(f'epochs: {epochs}') #! size
        logger.info(f'size: {size[0]}x{size[1]} (fullsize: {fullres})') #! size
        transform_str = 'transforms:\n'
        
        for t in transform_list[:-2]:
            transform_str += f'- {t}\n'
        logger.info(transform_str)

        input_size = dataset_class.width * dataset_class.height * dataset_class.channels

        # model = DummyNN(input_size, dataset_class.num_classes)
        #model = CNN(dataset_class.channels, dataset_class.num_classes)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = imitation_model.CNN(dataset_class.channels, dataset_class.num_classes).to(device)
        print(next(model.parameters()).device)
        # model = shufflenet_v2_x1_0(weights=ShuffleNet_V2_X1_0_Weights.DEFAULT)
        # model.fc = nn.Linear(model.fc.in_features, dataset_class.num_classes)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.003)
        # optimizer = optim.Adam(model.fc.parameters(), lr=0.003)
        model.train()
        predictions = [] #!
        labels = [] #!
        #!  time
        from time import time
        time_before_fit = time()
        #!/ time
        for epoch in range(epochs):
            # if epoch >= 3: #TODO is this useful? (idea: get fast good progress in first few epochs, then get better but slower progress w/ next epochs)
            #     #! basically, get good "nearly-initial" weights 
            #     dataset.transform = med_res_transform
            # elif epoch >= 6:
            #     dataset.transform = full_res_transform
            time_before_epoch = time() #! time
            loss_per_batch = []
            for _, (data, target) in enumerate(train_loader):
                #!  LR
                # if epoch == 10:
                #     for pg in optimizer.param_groups:
                #         pg['lr'] = 0.0006
                #!/ LR
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
            time_after_epoch = time() #! time
            timediff = time_after_epoch - time_before_epoch
            logger.info(f"Epoch {epoch + 1}, Loss: {np.mean(loss_per_batch)}, Time: {timediff // 60:.0f}m{timediff % 60:.3f}s")
            
			#!  accuracy & time
            from sklearn.metrics import accuracy_score
            if not np.isnan(labels).any():
                acc = accuracy_score(labels, predictions)
                logger.info(f"Accuracy on train set: {acc}\n")
                labels, predictions = [], []
            #!/ accuracy & time
        #!  time
        time_after_fit = time()
        timediff = time_after_fit - time_before_fit
        logger.info(f"{timediff // 60:.0f}m{timediff % 60:.3f}s")
        #!/ time
            
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
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        with torch.no_grad():
            for data, target in data_loader:
                data = data.to(device)
                target = target.to(device)
                output = self._model(data)
                predicted = torch.argmax(output, 1)
                
                predicted = predicted.to('cpu')
                target = target.to('cpu')
                
                labels.append(target.numpy())
                predictions.append(predicted.numpy())
        predictions = np.concatenate(predictions)
        labels = np.concatenate(labels)
        
        return predictions, labels

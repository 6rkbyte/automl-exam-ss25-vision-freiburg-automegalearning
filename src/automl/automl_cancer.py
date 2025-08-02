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
from sklearn.metrics import accuracy_score

from torch import nn, optim
from torch.utils.data import DataLoader
# from torchvision import transforms #!

#from automl.dummy_model import DummyNN, CNN
from automl.utils import calculate_mean_std

from automl import imitation_model
from automl import pretrained
import torchvision.transforms.v2 as transforms #!
import pandas as pd
from sklearn.model_selection import train_test_split
import optuna
from time import time

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
        hyperparams,
        trial: optuna.Trial | None = None,
        epochs=5,
        RESIZE_SIZE=0,
    ) -> AutoML:
        """A reference/toy implementation of a fitting function for the AutoML class.
        """
        # set seed for pytorch training
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed) #TODO should this be removed? should I just remove the hardcoded seed=42?
        torch.cuda.manual_seed(self.seed)

        # Ensure deterministic behavior in CuDNN
        torch.backends.cudnn.deterministic = True #TODO should this be removed?
        torch.backends.cudnn.benchmark = False

        res = min(dataset_class.width, RESIZE_SIZE) if RESIZE_SIZE != 0 else dataset_class.width #!
        res = (res, res) if isinstance(res, int) else res
        print(hyperparams)

        #!# IGNORE (removed the rest of the augment code to make this easier to navigate. keeping this little part as a reminder, though.)
        TRIVIAL = False #!#TODO use TrivialAugment?
        AUGMENT = True
        print(f'Using TRIVIAL Augment?: {TRIVIAL}')
        augment_list = []

        if not AUGMENT:
            transform_list = []
        else:
            transform_list = [a for a in augment_list]
        transforms_MAIN = [transforms.ToImage(), transforms.ToDtype(torch.float32, scale=True), transforms.Normalize(*calculate_mean_std(dataset_class))]
        transform_list += transforms_MAIN
        #!/ IGNORE

        #!  size; IGNORE
        fullres = (res[0] == dataset_class.width)
        if not fullres:
            transforms_MAIN = [transforms.Resize(res, interpolation=transforms.InterpolationMode.BICUBIC)] + transforms_MAIN
            transform_list = [transforms.Resize(res, interpolation=transforms.InterpolationMode.BICUBIC)] + transform_list
        transform_WITH_AUGMENTS = transforms.Compose(transform_list)
        self._transform = transforms.Compose(transforms_MAIN) #TODO separate; rebuild transforms in predict() instead of doing this
        
        size = (dataset_class.width, dataset_class.height) if fullres else res #! size
        print(f'epochs: {epochs}') #! size
        print(f'size: {size[0]}x{size[1]} (fullsize: {fullres})\n') #! size
		#!/ size; IGNORE

        #!# prints, IGNORE
        transform_str = 'transforms:\n'
        # for t in augment_list:
        for t in augment_list:
            transform_str += f'- {t}\n'
        print(transform_str)
        if augment_list:
            augment_list = transforms.Compose(augment_list)
        #!/ prints, IGNORE

        dataset = dataset_class(
            root="./data",
            split='train',
            download=True,
            transform=transform_WITH_AUGMENTS
        )

        #TODO n_workers train_loader
        num_workers = 4 if size[0] <= 256 else 2 #TODO remove num_workers? probably good tho. can their PCs use it? probably
        train_loader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=num_workers, pin_memory=True)
        input_size = dataset_class.width * dataset_class.height * dataset_class.channels

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # INITIAL_LR = 0.003
        # INITIAL_LR = 0.005
        INITIAL_LR = hyperparams['initial_lr']
        LR_DECAY = hyperparams['lr_decay']
        DROPOUT = hyperparams['dropout']
        NETWORK = hyperparams['network']
        #! I use a default dict to allow for easy code changes. This is here as a guard for when I'm not searching for these hyperparams,
        #! e.g. choose_network(): calls .fit() with only <network_name> as hyperparam; use default values for these.
        if INITIAL_LR == 0 and LR_DECAY == 0 and DROPOUT == 0:
            INITIAL_LR = 0.003
            LR_DECAY = 0.2
            DROPOUT = 0.2

        if NETWORK == 'pretrained':
            model = pretrained.Mobilenet(input_shape=None, num_classes=dataset_class.num_classes, dropout=DROPOUT).to(device)
        else:
            model = imitation_model.CNN(dataset_class.channels, dataset_class.num_classes).to(device)
        self._model = model
        # model = pretrained.Vit(input_shape=dataset_class.channels, num_classes=dataset_class.num_classes).to(device)

        labels = torch.tensor([label for _, label in dataset]).to(device) #!
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=INITIAL_LR)
        # optimizer = optim.Adam(model.vit.head.parameters(), lr=INITIAL_LR)
        # optimizer = optim.AdamW(model.parameters(), lr=INITIAL_LR, weight_decay=1e-2) #! other regularization options
        # optimizer = optim.Adam(model.fc.parameters(), lr=0.003)
        model.train()
        predictions = []
        labels = []
        time_before_fit = time() #! time
        print('\nStarting Training')

        #! currently unused. 
        TRAIN_BACKBONE_INITIALLY = False # train flowers: 5 normal, 5 backbone (mby optuna: tune_backbone?, LRs) 
                                        # if small search space: grid search; if fast searching: large search space
        TRAIN_BACKBONE_LATER = False
        print(f'Initial LR: {INITIAL_LR}\nLR Decay: {LR_DECAY}\nDropout: {DROPOUT}')
        # print(f'Training backbone:\n- Initially: {TRAIN_BACKBONE_INITIALLY}\n- Later: {TRAIN_BACKBONE_LATER}\n')
        # model.train_backbone(TRAIN_BACKBONE_INITIALLY) #!
        for epoch in range(epochs):
            time_before_epoch = time() #! time
            loss_per_batch = []
            # if dataset_class._dataset_name == 'skin_cancer':
            #     # lr_decrease_epoch = 4
            #     lr_decrease_epoch = 5
            # elif dataset_class._dataset_name == 'flowers':
            #     lr_decrease_epoch = 5
            # else:
            #     lr_decrease_epoch = 5
            lr_decrease_epoch = 5 #!#TODO choose this?
            if epoch == lr_decrease_epoch:
                # model.train_backbone(TRAIN_BACKBONE_LATER) #!
                for pg in optimizer.param_groups:
                    pg['lr'] *= LR_DECAY
                    # pg['lr'] = 0.0006
                    # pg['lr'] *= 0.1

            for _, (data, target) in enumerate(train_loader):
                data = data.to(device)
                target = target.to(device)
                
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                loss_per_batch.append(loss.item())

                #!# accuracy
                predicted = torch.argmax(output, 1)   
                target = target.to('cpu')
                predicted = predicted.to('cpu')
                predictions.append(predicted.numpy())
                labels.append(target.numpy())        
                #!/ accuracy
            predictions = np.concatenate(predictions)
            labels = np.concatenate(labels)
            
            #!# time
            time_after_epoch = time()
            timediff = time_after_epoch - time_before_epoch
            #logger.info(f"Epoch {epoch + 1}, Loss: {np.mean(loss_per_batch)}, Time: {timediff // 60:.0f}m{timediff % 60:.3f}s") #!
            print(f"Epoch {epoch + 1}, Loss: {np.mean(loss_per_batch)}, Time: {timediff // 60:.0f}m{timediff % 60:.3f}s\n")
            #!/ time
			#!  accuracy
            if not np.isnan(labels).any():
                acc = accuracy_score(labels, predictions)
                #logger.info(f"Accuracy on train set: {acc}\n") #!
                print(f"Accuracy on train set: {acc}")
                labels, predictions = [], []
            #!/ accuracy
            
            # model.eval()
            # val_predictions, val_labels = self.predict(dataset_class, val_loader)
            # val_acc = accuracy_score(val_labels, val_predictions)
            # trial.report(val_acc, epoch)
            # if trial.should_prune():
            #     raise optuna.exceptions.TrialPruned()
            # model.train()

        #!  time
        time_after_fit = time()
        timediff = time_after_fit - time_before_fit
        #logger.info(f"{timediff // 60:.0f}m{timediff % 60:.3f}s") #!
        print(f"{timediff // 60:.0f}m{timediff % 60:.3f}s")
        #!/ time
            

        return self

    def predict(self, dataset_class, data_loader: DataLoader | None = None) -> Tuple[np.ndarray, np.ndarray]:
        """A reference/toy implementation of a prediction function for the AutoML class.
        """
        split = 'val' if dataset_class._dataset_name == 'skin_cancer' else 'test'
        dataset = dataset_class(
            root="./data",
            split=split,
            download=True,
            transform=self._transform
        )
        if data_loader == None:
            data_loader = DataLoader(dataset, batch_size=100, shuffle=False, pin_memory=True)
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

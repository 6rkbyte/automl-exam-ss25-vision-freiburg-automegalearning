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
# from torchvision import transforms #!

#from automl.dummy_model import DummyNN, CNN
from automl.utils import calculate_mean_std

from automl import imitation_model
import torchvision.transforms.v2 as transforms #!
import pandas as pd
from sklearn.model_selection import train_test_split

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
        augments,
        epochs=5,
        RESIZE_SIZE=0
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
        print(augments)
        # w = 4*res[0]/dataset_class.width #!#TODO normalization for blur
        w=1 #!
        TRIVIAL = False #!#TODO use TrivialAugment?
        AUGMENT = True
        print(f'Using TRIVIAL Augment?: {TRIVIAL}')
        augment_list = []

        #!# Default (Manually choose Trivial, or Others)
        if not TRIVIAL and AUGMENT:
            if (rot := augments['rot']) != 0:
                augment_list.append(transforms.RandomRotation(degrees=rot, interpolation=transforms.InterpolationMode.BICUBIC))
                # augment_list.append(transforms.RandomRotation(degrees=rot, interpolation=transforms.InterpolationMode.BILINEAR))
            if (horflip := augments['horflip']) != 0:
                augment_list.append(transforms.RandomHorizontalFlip(horflip))
                # transform_list.append(transforms.RandomHorizontalFlip())
            if (verflip := augments['verflip']) != 0:
                augment_list.append(transforms.RandomVerticalFlip(verflip))
                # transform_list.append(transforms.RandomVerticalFlip())
            if (blurstddev := augments['blur']) != 0:
                # transform_list.append(transforms.GaussianBlur(kernel_size=(5,5), sigma=(0.1, blurstddev)))
                #!#TODO normalize by imagesize (*w)
                augment_list.append((transforms.RandomApply(
                    [transforms.GaussianBlur(kernel_size=(3,3), sigma=(0.1*w, blurstddev*w))
                    ], p=0.3)))
            if (noisestddev := augments['noise']) != 0:
                # transform_list.append(transforms.GaussianNoise(sigma=noisestddev))
                #transform_list.append(transforms.GaussianNoise(sigma=noisestddev))
                augment_list.append(transforms.RandomApply([transforms.GaussianNoise(sigma=noisestddev)], p=0.3))
            if (elastic := augments['elastic']) != 0:
                augment_list.append(transforms.RandomApply([transforms.ElasticTransform()], p=elastic))
            if (gray := augments['gray']) != 0:
                augment_list.append(transforms.RandomApply([transforms.Grayscale(dataset_class.channels)], p=gray))
            if (affine := augments['affine']) != 0:
                augment_list.append(transforms.RandomApply([transforms.RandomAffine(degrees=affine)], p=0.5))
        elif TRIVIAL:
            trivial_augment = transforms.TrivialAugmentWide(interpolation=transforms.InterpolationMode.BICUBIC)
            # trivial_augment = transforms.AugMix(interpolation=transforms.InterpolationMode.BILINEAR)
            augment_list.append(trivial_augment)
        #!/ Default (Manually choose Trivial, or Others)

        #!# RandomChoice(Trivial, Others)
        # if (rot := augments['rot']) != 0:
        #     augment_list.append(transforms.RandomRotation(degrees=rot, interpolation=transforms.InterpolationMode.BICUBIC))
        #     # augment_list.append(transforms.RandomRotation(degrees=rot, interpolation=transforms.InterpolationMode.BILINEAR))
        # if (horflip := augments['horflip']) != 0:
        #     augment_list.append(transforms.RandomHorizontalFlip(horflip))
        #     # transform_list.append(transforms.RandomHorizontalFlip())
        # if (verflip := augments['verflip']) != 0:
        #     augment_list.append(transforms.RandomVerticalFlip(verflip))
        #     # transform_list.append(transforms.RandomVerticalFlip())
        # if (blurstddev := augments['blur']) != 0:
        #     # transform_list.append(transforms.GaussianBlur(kernel_size=(5,5), sigma=(0.1, blurstddev)))
        #     #!#TODO normalize by imagesize (*w)
        #     augment_list.append((transforms.RandomApply(
        #         [transforms.GaussianBlur(kernel_size=(3,3), sigma=(0.1*w, blurstddev*w))
        #         ], p=0.3)))
        # if (noisestddev := augments['noise']) != 0:
        #     # transform_list.append(transforms.GaussianNoise(sigma=noisestddev))
        #     #transform_list.append(transforms.GaussianNoise(sigma=noisestddev))
        #     augment_list.append(transforms.RandomApply([transforms.GaussianNoise(sigma=noisestddev)], p=0.3))
        # if (elastic := augments['elastic']) != 0:
        #     augment_list.append(transforms.RandomApply([transforms.ElasticTransform()], p=elastic))
        # if (gray := augments['gray']) != 0:
        #     augment_list.append(transforms.RandomApply([transforms.Grayscale(dataset_class.channels)], p=gray))
        # if (affine := augments['affine']) != 0:
        #     augment_list.append(transforms.RandomApply([transforms.RandomAffine(degrees=affine)], p=0.5))
        # trivial_augment = transforms.TrivialAugmentWide(interpolation=transforms.InterpolationMode.BICUBIC)
        # # augment_list.append(trivial_augment)
        # augment_list = [transforms.RandomChoice([transforms.Compose(augment_list), trivial_augment])]
        #!/ RandomChoice(Trivial, Others)


        # transform_list = []

        # # if transform_list != []:
        # #     transform_list = [transforms.RandomChoice([t for t in transform_list], p=None)] #TODO RandomChoice
        # transform_list += [transforms.ToImage(), transforms.ToDtype(torch.float32, scale=True)]
        # transform_list += [
        #     #transforms.ToTensor(),
        #     #transforms.ToImage(), transforms.ToDtype(torch.float32, scale=True), # <=> ToTensor()
        #     transforms.Normalize(*calculate_mean_std(dataset_class)),
        # ]
        if not AUGMENT:
            transform_list = []
        else:
            transform_list = [a for a in augment_list]
        # transform_list_MAIN += augment_list
        transforms_MAIN = [transforms.ToImage(), transforms.ToDtype(torch.float32, scale=True), transforms.Normalize(*calculate_mean_std(dataset_class))]
        transform_list += transforms_MAIN
        # transform_list_MAIN += [transforms.ToImage(), transforms.ToDtype(torch.float32, scale=True)]
        #augment_list.append(transforms.Normalize(*calculate_mean_std(dataset_class)))

        #!  size
        #res = 0 # 0 (fullres), 84, 112, 224
        #res = 56
        fullres = (res[0] == dataset_class.width)
        if not fullres:
            transforms_MAIN = [transforms.Resize(res, interpolation=transforms.InterpolationMode.BICUBIC)] + transforms_MAIN
            transform_list = [transforms.Resize(res, interpolation=transforms.InterpolationMode.BICUBIC)] + transform_list
        transform_WITH_AUGMENTS = transforms.Compose(transform_list)
        #self._transform = full_res_transform if fullres else transform
        self._transform = transforms.Compose(transforms_MAIN) #TODO separate; rebuild transforms in predict() instead of doing this
		#!/ size
        #epochs = 10 #!
        #train_loader = DataLoader(dataset, batch_size=64, shuffle=True, pin_memory=True)
        size = (dataset_class.width, dataset_class.height) if fullres else res #! size
        print(f'epochs: {epochs}') #! size
        print(f'size: {size[0]}x{size[1]} (fullsize: {fullres})\n') #! size
        transform_str = 'transforms:\n'
        
        # for t in augment_list:
        for t in transform_list:
            transform_str += f'- {t}\n'
        print(transform_str)
        transform_str = 'augments:\n'
        
        # for t in augment_list:
        for t in augment_list:
            transform_str += f'- {t}\n'
        print(transform_str)
        if augment_list:
            augment_list = transforms.Compose(augment_list)
        dataset = dataset_class(
            root="./data",
            split='train',
            download=True,
            transform=transform_WITH_AUGMENTS
        )

        #TODO n_workers train_loader
        # train_loader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=3, pin_memory=True, generator=torch.Generator().manual_seed(42))
        train_loader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=4, pin_memory=True)
        input_size = dataset_class.width * dataset_class.height * dataset_class.channels

        # model = DummyNN(input_size, dataset_class.num_classes)
        #model = CNN(dataset_class.channels, dataset_class.num_classes)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = imitation_model.CNN(dataset_class.channels, dataset_class.num_classes).to(device)
        # model = shufflenet_v2_x1_0(weights=ShuffleNet_V2_X1_0_Weights.DEFAULT)
        # model.fc = nn.Linear(model.fc.in_features, dataset_class.num_classes)
        # labels = torch.tensor([label for _, label in dataset]).to(device) #!
        # class_counts = torch.bincount(labels)                             #!
        # class_weights = 1.0 / class_counts.float()                        #!
        # criterion = nn.CrossEntropyLoss(weight=class_weights)             #!
        
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
        print('starting SGD')

        for epoch in range(epochs):
            # if epoch >= 3: #TODO is this useful? (idea: get fast good progress in first few epochs, then get better but slower progress w/ next epochs)
            #     #! basically, get good "nearly-initial" weights 
            #     dataset.transform = med_res_transform
            # elif epoch >= d6:
            #     dataset.transform = full_res_transform
            time_before_epoch = time() #! time
            loss_per_batch = []
            for _, (data, target) in enumerate(train_loader):
                data = data.to(device)
                # data = augment_list(data)
                #!  LR
                #if epoch == 10:
                #TODO
                # previously: 4, then 10
                # lr_decrease_epoch = 10 if dataset_class._dataset_name != 'skin_cancer' else 4
                # lr_decrease_epoch = 5 if dataset_class._dataset_name != 'skin_cancer' else 4
                if dataset_class._dataset_name == 'skin_cancer':
                    lr_decrease_epoch = 4
                elif dataset_class._dataset_name == 'flowers':
                    lr_decrease_epoch = 10
                else:
                    lr_decrease_epoch = 5
                if epoch == lr_decrease_epoch:
                    for pg in optimizer.param_groups:
                        pg['lr'] = 0.0006
                # elif epoch == 10: #TODO LR
                #     for pg in optimizer.param_groups:
                #         pg['lr'] = 0.00003
                #!/ LR
                # data = data.to(device)
                target = target.to(device)
                
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                loss_per_batch.append(loss.item())

                predicted = torch.argmax(output, 1)   #! accuracy
                
                target = target.to('cpu')
                predicted = predicted.to('cpu')
                
                predictions.append(predicted.numpy()) #! accuracy
                labels.append(target.numpy())         #! accuracy
            predictions = np.concatenate(predictions)
            labels = np.concatenate(labels)
            time_after_epoch = time() #! time
            timediff = time_after_epoch - time_before_epoch
            #logger.info(f"Epoch {epoch + 1}, Loss: {np.mean(loss_per_batch)}, Time: {timediff // 60:.0f}m{timediff % 60:.3f}s") #!
            print(f"Epoch {epoch + 1}, Loss: {np.mean(loss_per_batch)}, Time: {timediff // 60:.0f}m{timediff % 60:.3f}s")
            
			#!  accuracy & time
            from sklearn.metrics import accuracy_score
            if not np.isnan(labels).any():
                acc = accuracy_score(labels, predictions)
                #logger.info(f"Accuracy on train set: {acc}\n") #!
                print(f"Accuracy on train set: {acc}\n")
                labels, predictions = [], []
            #!/ accuracy & time
        #!  time
        time_after_fit = time()
        timediff = time_after_fit - time_before_fit
        #logger.info(f"{timediff // 60:.0f}m{timediff % 60:.3f}s") #!
        print(f"{timediff // 60:.0f}m{timediff % 60:.3f}s")
        #!/ time
            
        model.eval()
        self._model = model

        return self

    def predict(self, dataset_class) -> Tuple[np.ndarray, np.ndarray]:
        """A reference/toy implementation of a prediction function for the AutoML class.
        """
        split = 'val' if dataset_class._dataset_name == 'skin_cancer' else 'test'
        dataset = dataset_class(
            root="./data",
            split=split,
            download=True,
            transform=self._transform
        )
        data_loader = DataLoader(dataset, batch_size=100, shuffle=False, pin_memory=True, num_workers=2) #TODO remove num_workers? probably good tho. can their PCs use it? probably
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

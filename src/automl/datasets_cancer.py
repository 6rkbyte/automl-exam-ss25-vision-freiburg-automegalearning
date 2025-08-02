"""
This module contains the datasets used in the AutoML exam.
If you want to edit this file be aware that we will later 
  push the test set to this file which might cause problems.
"""

from pathlib import Path
from typing import Any, Callable, Optional, Tuple, Union

import PIL.Image
import pandas as pd
from torchvision.datasets import VisionDataset
from torchvision.datasets.utils import download_and_extract_archive, check_integrity
import tempfile
import shutil
from torch import tensor
from sklearn.model_selection import train_test_split


class BaseVisionDataset(VisionDataset):
    """A base class for all vision datasets.

    Args:
        root: str or Path
            Root directory of the dataset (should contain the extracted datasets).
        split: string (optional)
            The dataset split, supports `train` (default), `val`, or `test`.
        transform: callable (optional)
            A function/transform that takes in a PIL image and returns a transformed version. 
            E.g, `transforms.RandomCrop`.
        target_transform: callable (optional)
            A function/transform that takes in the target and transforms it.
        download: bool (optional)
            If true, downloads the dataset zip and extracts it into the root directory.
            If dataset is already downloaded, it is not downloaded again.
    """
  
    _dataset_name: str
    width: int
    height: int
    channels: int
    num_classes: int

    PHASE1_URL = "https://ml.informatik.uni-freiburg.de/research-artifacts/automl-exam-25-vision/vision-phase1.zip "
    PHASE2_URL = "https://ml.informatik.uni-freiburg.de/research-artifacts/automl-exam-25-vision/vision-phase2.zip "

    def __init__(
        self,
        root: Union[str, Path] = "data",
        split: str = "train",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ) -> None:
        print([t for t in transform.transforms])
        super().__init__(root, transform=transform, target_transform=target_transform)
        assert split in ["train", "test"], f"Split {split} not supported"
        self._split = split
        self._base_folder = Path(self.root) / self._dataset_name

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError(
                "Dataset not found or corrupted. Use download=True to download required files, "
                "or download them manually from https://ml.informatik.uni-freiburg.de/research-artifacts/automl-exam-25-vision/"
            )

        data = pd.read_csv(self._base_folder / f"{split}.csv")
        self._labels = data['label'].tolist()
        self._image_files = data['image_file_name'].tolist()

    def _check_integrity(self) -> bool:
        train_images_folder = self._base_folder / "images_train"
        test_images_folder = self._base_folder / "images_test"
        if not (train_images_folder.exists() and train_images_folder.is_dir()) or \
           not (test_images_folder.exists() and test_images_folder.is_dir()):
            return False
        if not (self._base_folder / "train.csv").exists() or not (self._base_folder / "test.csv").exists():
            return False
        return True

    def download(self) -> None:
        """Download dataset if missing, choosing phase1 or phase2 based on dataset name."""
        data_path = Path(self.root)
        data_path.mkdir(exist_ok=True)

        if self._base_folder.exists():
            # print(f"{self._dataset_name} dataset already exists. Skipping download.")
            return

        if self._dataset_name == "skin_cancer":
            phase_url = self.PHASE2_URL
            phase_name = "phase2"
        else:
            phase_url = self.PHASE1_URL
            phase_name = "phase1"

        with tempfile.TemporaryDirectory() as temp_dir:
            print(f"Downloading and extracting {phase_name} dataset...")
            zip_name = f"{phase_name}.zip"

            download_and_extract_archive(
                url=phase_url,
                download_root=temp_dir,
                extract_root=temp_dir,
                filename=zip_name,
                remove_finished=True,
            )

            phase_folder = next(Path(temp_dir).glob("phase*"))
            for item in phase_folder.iterdir():
                destination = data_path / item.name
                if destination.exists():
                    print(f"Skipping existing item: {destination}")
                    continue
                shutil.move(str(item), str(destination))

            print(f"{phase_name} download completed.")

    def extra_repr(self) -> str:
        """String representation of the dataset."""
        return f"split={self._split}"

    def __getitem__(self, idx: int) -> Tuple[Any, Any]:
        image_file, label = self._image_files[idx], self._labels[idx]
        image_path = self._base_folder / f"images_{self._split}" / image_file
        image = PIL.Image.open(image_path)
        if self.channels == 1:
            image = image.convert("L")
        elif self.channels == 3:
            image = image.convert("RGB")
        else:
            raise ValueError(f"Unsupported number of channels: {self.channels}")

        if self.transform:
            image = self.transform(image)

        if self.target_transform:
            label = self.target_transform(label)

        return image, label

    def __len__(self) -> int:
        return len(self._image_files)



class EmotionsDataset(BaseVisionDataset):
    """ Emotions Dataset.

    "This dataset contains images of faces displaying one of seven emotions
    (0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral).
    """
    _dataset_name = "emotions"
    width = 48
    height = 48
    channels = 1
    num_classes = 7
    mean = tensor([0.5077]) #TODO REMOVE
    std = tensor([0.2118]) #TODO REMOVE


class FlowersDataset(BaseVisionDataset):
    """Flower Dataset.

    This dataset contains images of 102 types of flowers. The task is to classify the flower type.
    """
    _dataset_name = "flowers"
    width = 512
    height = 512
    channels = 3
    num_classes = 102
    mean = tensor([0.4341, 0.3762, 0.2862]) #TODO REMOVE
    std = tensor([0.2645, 0.2118, 0.2182]) #TODO REMOVE


class FashionDataset(BaseVisionDataset):
    """Fashion Dataset.

    This dataset contains images of fashion items. The task is to classify what kind of fashion item it is.
    """
    _dataset_name = "fashion"
    width = 28
    height = 28
    channels = 1
    num_classes = 10
    mean=tensor([0.2889]) #TODO REMOVE
    std=tensor([0.3180]) #TODO REMOVE



class BaseCancerDataset(VisionDataset):
    """A base class for all vision datasets.

    Args:
        root: str or Path
            Root directory of the dataset (should contain the extracted datasets).
        split: string (optional)
            The dataset split, supports `train` (default), `val`, or `test`.
        transform: callable (optional)
            A function/transform that takes in a PIL image and returns a transformed version. 
            E.g, `transforms.RandomCrop`.
        target_transform: callable (optional)
            A function/transform that takes in the target and transforms it.
        download: bool (optional)
            If true, downloads the dataset zip and extracts it into the root directory.
            If dataset is already downloaded, it is not downloaded again.
    """
  
    _dataset_name: str
    width: int
    height: int
    channels: int
    num_classes: int

    PHASE1_URL = "https://ml.informatik.uni-freiburg.de/research-artifacts/automl-exam-25-vision/vision-phase1.zip "
    PHASE2_URL = "https://ml.informatik.uni-freiburg.de/research-artifacts/automl-exam-25-vision/vision-phase2.zip "

    def __init__(
        self,
        root: Union[str, Path] = "data",
        split: str = "train",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ) -> None:
        print([t for t in transform.transforms])
        super().__init__(root, transform=transform, target_transform=target_transform)
        assert split in ["train", "test", "val"], f"Split {split} not supported"
        split_original = "test" if split == "test" else "train"
        self._split = split_original
        self._base_folder = Path(self.root) / self._dataset_name

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError(
                "Dataset not found or corrupted. Use download=True to download required files, "
                "or download them manually from https://ml.informatik.uni-freiburg.de/research-artifacts/automl-exam-25-vision/"
            )

        data = pd.read_csv(self._base_folder / f"{split_original}.csv")
        if split == "test":
            data = pd.read_csv(self._base_folder / "test.csv")
            self._labels = data['label'].tolist()
            self._image_files = data['image_file_name'].tolist()
        else:
            data = pd.read_csv(self._base_folder / "train.csv")
            train_data, val_data = train_test_split(data, test_size=0.25, random_state=42, stratify=data['label'])
            if split == "train":
                self._labels = train_data['label'].tolist()
                self._image_files = train_data['image_file_name'].tolist()
            else: # val
                self._labels = val_data['label'].tolist()
                self._image_files = val_data['image_file_name'].tolist()

    def _check_integrity(self) -> bool:
        train_images_folder = self._base_folder / "images_train"
        test_images_folder = self._base_folder / "images_test"
        if not (train_images_folder.exists() and train_images_folder.is_dir()) or \
           not (test_images_folder.exists() and test_images_folder.is_dir()):
            return False
        if not (self._base_folder / "train.csv").exists() or not (self._base_folder / "test.csv").exists():
            return False
        return True

    def download(self) -> None:
        """Download dataset if missing, choosing phase1 or phase2 based on dataset name."""
        data_path = Path(self.root)
        data_path.mkdir(exist_ok=True)

        if self._base_folder.exists():
            # print(f"{self._dataset_name} dataset already exists. Skipping download.")
            return

        if self._dataset_name == "skin_cancer":
            phase_url = self.PHASE2_URL
            phase_name = "phase2"
        else:
            phase_url = self.PHASE1_URL
            phase_name = "phase1"

        with tempfile.TemporaryDirectory() as temp_dir:
            print(f"Downloading and extracting {phase_name} dataset...")
            zip_name = f"{phase_name}.zip"

            download_and_extract_archive(
                url=phase_url,
                download_root=temp_dir,
                extract_root=temp_dir,
                filename=zip_name,
                remove_finished=True,
            )

            phase_folder = next(Path(temp_dir).glob("phase*"))
            for item in phase_folder.iterdir():
                destination = data_path / item.name
                if destination.exists():
                    print(f"Skipping existing item: {destination}")
                    continue
                shutil.move(str(item), str(destination))

            print(f"{phase_name} download completed.")

    def extra_repr(self) -> str:
        """String representation of the dataset."""
        return f"split={self._split}"

    def __getitem__(self, idx: int) -> Tuple[Any, Any]:
        image_file, label = self._image_files[idx], self._labels[idx]
        image_path = self._base_folder / f"images_{self._split}" / image_file
        image = PIL.Image.open(image_path)
        if self.channels == 1:
            image = image.convert("L")
        elif self.channels == 3:
            image = image.convert("RGB")
        else:
            raise ValueError(f"Unsupported number of channels: {self.channels}")

        if self.transform:
            image = self.transform(image)

        if self.target_transform:
            label = self.target_transform(label)

        return image, label

    def __len__(self) -> int:
        return len(self._image_files)







class SkinCancerDataset(BaseCancerDataset):
    """SkinCancer Dataset.
    
    The SkinCancer dataset contains images of skin lesions. The task is to classify what kind of skin lesion it is.

    This is the test dataset for the AutoML exam. It does not contain the labels for the test split.
    You are expected to predict these labels and save them to a file called `final_test_preds.npy` for your
    final submission.
    """
    _dataset_name = "skin_cancer"
    width = 450
    height = 450
    channels = 3
    num_classes = 7
    mean=tensor([0.7611, 0.5442, 0.5693]) #TODO REMOVE
    std=tensor([0.0901, 0.1184, 0.1326]) #TODO REMOVE




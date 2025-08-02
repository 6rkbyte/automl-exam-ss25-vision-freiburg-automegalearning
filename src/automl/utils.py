from typing import Any

from torch.utils.data import DataLoader
from torchvision import transforms

from time import time


def calculate_mean_std(dataset_class: Any):
    """Calculate the mean and standard deviation of the entire image dataset."""
    mean = 0.
    std = 0.
    total_images_count = 0
    return dataset_class.mean, dataset_class.std #TODO REMOVE

    dataset = dataset_class(
        root="./data",
        split='train',
        download=True,
        transform=transforms.ToTensor() #TODO
    )
    loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=2)
    t1 = time()
    for images, _ in loader:
        batch_samples = images.size(0)  # batch size (the last batch can have smaller size!)
        images = images.view(batch_samples, images.size(1), -1)
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)
        total_images_count += batch_samples
    t2 = time()

    mean /= total_images_count
    std /= total_images_count
    print(f'\ntime to calculate mean_std (utils.py): {t2-t1}s\nmean: {mean}, std: {std}\n') #TODO REMOVE

    return mean, std

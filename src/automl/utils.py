from typing import Any

from torch.utils.data import DataLoader
from torchvision import transforms

import csv


def calculate_mean_std(dataset_class: Any):
    """Calculate the mean and standard deviation of the entire image dataset."""
    mean = 0.
    std = 0.
    total_images_count = 0

    dataset = dataset_class(
        root="./data",
        split='train',
        download=True,
        transform=transforms.ToTensor()
    )
    loader = DataLoader(dataset, batch_size=64, shuffle=False)

    for images, _ in loader:
        batch_samples = images.size(0)  # batch size (the last batch can have smaller size!)
        images = images.view(batch_samples, images.size(1), -1)
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)
        total_images_count += batch_samples

    mean /= total_images_count
    std /= total_images_count

    return mean, std


def init_log_file(filepath, headers):
    """Create a new CSV log file with column headers."""
    with open(filepath, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(headers)

def log_epoch_results(filepath, values):
    """Append a row of values (e.g., one epoch's results) to the CSV log file."""
    with open(filepath, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(values)
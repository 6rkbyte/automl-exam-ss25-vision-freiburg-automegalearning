from typing import Any

from torch.utils.data import DataLoader
from torchvision import transforms


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

class TrainLog:
    """
    Just a list to store the loss and time.
    """

    def __init__(self):
        # {"metric_name": [x_list, y_list], ...}
        self.data = {}

    def append(self, metric: str, x: float, y: float):
        if metric not in self.data:
            self.data[metric] = [[], []] # [x_list, y_list]

        self.data[metric][0].append(x) # x_list
        self.data[metric][1].append(y) # y_list

def plot_loss(log: TrainLog):
    """
    Plot value array y over time t.
    """

    fig, ax = plt.subplots(1, 1, figsize=(6, 3))

    t = log.data['train'][0]
    y = log.data['train'][1]
    ax.plot(t, y, color='b', label='train')

    t = log.data['val'][0]
    y = log.data['val'][1]
    ax.plot(t, y, color='g', label='val')

    ax.set_title('Loss')
    ax.set_xlabel('Epoch')
    #ax.set_xticks(np.arange(t[0], t[-1]+1, len(t)//10+1))
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig("loss_curve.png")
    plt.show()

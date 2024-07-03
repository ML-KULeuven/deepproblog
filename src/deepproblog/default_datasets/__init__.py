from pathlib import Path

import torchvision
import torchvision.transforms as transforms

_DATA_ROOT = Path(__file__).parent

mnist_transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
)


class ImageDataset(object):
    def __init__(self, dataset):
        self.dataset = dataset

    def __getitem__(self, item):
        index = int(item[0])
        if index < 0:
            return -self.dataset[-index][0]
        else:
            return self.dataset[index][0]


default_datasets = {
    "mnist_train": ImageDataset(
        torchvision.datasets.MNIST(
            root=str(_DATA_ROOT), train=True, download=True, transform=mnist_transform
        )
    ),
    "mnist_test": ImageDataset(
        torchvision.datasets.MNIST(
            root=str(_DATA_ROOT), train=False, download=True, transform=mnist_transform
        )
    ),
}

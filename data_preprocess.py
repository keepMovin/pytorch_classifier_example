import torch
import torch.nn as nn
from torchvision import datasets
from torchvision import transforms, utils
from torch.utils.data import Dataset, DataLoader
from torch.utils import data


def get_transforms(train=False):
    transform = []
    if train:
        transform.append(transforms.RandomHorizontalFlip(0.5))
    transform.append(transforms.ToTensor())
    return transforms.Compose(transform)

image_data = datasets.ImageFolder('data/shapes/', get_transforms(train=True))
image_data_test = datasets.ImageFolder('data/test_shapes/', get_transforms(train=False))

print(len(image_data_test))

# train_dataloader = data.DataLoader(image_data, batch_size=32, shuffle=True)
# test_dataloader = data.DataLoader(image_data_test, batch_size=32, shuffle=True)
#
# print(len(train_dataloader))



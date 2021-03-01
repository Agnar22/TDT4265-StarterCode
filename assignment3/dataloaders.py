from torchvision import transforms, datasets
from torch.utils.data.sampler import SubsetRandomSampler
import torch
import typing
import numpy as np
np.random.seed(0)

mean = (0.485, 0.456, 0.406)
std = (.229, .224, .225)


def load_cifar10(batch_size: int, validation_fraction: float = 0.1, resize: bool = False,
                 augmentation: bool = False) -> typing.List[torch.utils.data.DataLoader]:
    # Note that transform train will apply the same transform for
    # validation!
    if(resize):
        transform_train = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        transform_test = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    else:
        if augmentation:
          transform_train = transforms.Compose([
              transforms.ColorJitter(brightness=0.2, contrast=0.0, saturation=0.2, hue=0.00),
              transforms.RandomHorizontalFlip(),
              transforms.ToTensor(),
              transforms.Normalize(mean, std)
          ])
        else:
          transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
          ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    data_train = datasets.CIFAR10('data/cifar10',
                                  train=True,
                                  download=True,
                                  transform=transform_train)

    data_val = datasets.CIFAR10('data/cifar10',
                                  train=True,
                                  download=True,
                                  transform=transform_test)

    data_test = datasets.CIFAR10('data/cifar10',
                                 train=False,
                                 download=True,
                                 transform=transform_test)

    indices = list(range(len(data_train)))
    split_idx = int(np.floor(validation_fraction * len(data_train)))

    val_indices = np.random.choice(indices, size=split_idx, replace=False)
    train_indices = list(set(indices) - set(val_indices))

    train_sampler = SubsetRandomSampler(train_indices)
    validation_sampler = SubsetRandomSampler(val_indices)

    dataloader_train = torch.utils.data.DataLoader(data_train,
                                                   sampler=train_sampler,
                                                   batch_size=batch_size,
                                                   num_workers=2,
                                                   drop_last=True)

    dataloader_val = torch.utils.data.DataLoader(data_val,
                                                 sampler=validation_sampler,
                                                 batch_size=batch_size,
                                                 num_workers=2)

    dataloader_test = torch.utils.data.DataLoader(data_test,
                                                  batch_size=batch_size,
                                                  shuffle=False,
                                                  num_workers=2)

    return dataloader_train, dataloader_val, dataloader_test

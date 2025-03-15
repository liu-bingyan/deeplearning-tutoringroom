import numpy as np
import torch
import torchvision

from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

def load_data(dataset_name = 'MNIST', batch_size = 64, test_size = 0.2, random_state = 42, shuffle = True, normalize = True):
    if normalize:
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    else:
        transform = transforms.Compose([transforms.ToTensor()])

    if dataset_name == 'MNIST':
        # Define a transform to normalize the data
        trainset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    elif dataset_name == 'CelebA':
        # Download and load the training data
        trainset = datasets.CelebA(root='./data', split='train', download=True, transform=transform)
    elif dataset_name == 'CIFAR10':
        # Download and load the training data
        trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    else:
        raise ValueError('Dataset not supported')
    
    # Split the dataset into training and test sets
    train_data, test_data = train_test_split(trainset, test_size=test_size, random_state=random_state)

    # Create data loaders for training and test sets
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=shuffle)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False)
    return trainloader, testloader
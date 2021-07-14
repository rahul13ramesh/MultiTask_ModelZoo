#!/usr/bin/env python3
"""
Helper function to fetch a dataset
"""
from datasets.mnist import SplitMNISTHandler
from datasets.mnist import RotatedMNISTHandler
from datasets.mnist import PermutedMNISTHandler
from datasets.cifar import Cifar100Handler
from datasets.cifar import Cifar10Handler
from datasets.data import MultiTaskDataHandler


def fetch_dataclass(dataset: str) -> MultiTaskDataHandler:
    if dataset == "cifar100":
        return Cifar100Handler
    elif dataset == "cifar10":
        return Cifar10Handler
    elif dataset == "mnist":
        return SplitMNISTHandler
    elif dataset == "rotmnist":
        return RotatedMNISTHandler
    elif dataset == "permnist":
        return PermutedMNISTHandler
    else:
        raise ValueError("Invalid Dataset")

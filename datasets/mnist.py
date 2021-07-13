from typing import List

import numpy as np
import torchvision
import torchvision.transforms as transforms

from numpy.random import default_rng
from datasets.modmnist import ModMNIST
from datasets.data import MultiTaskDataHandler


class SplitMNISTHandler(MultiTaskDataHandler):
    """
    Load SplitMNIST dataset Split 10 classes into multiple tasks
    """
    def __init__(self,
                 tasks: List[List[int]],
                 samples: int,
                 seed: int = -1) -> None:
        """
        Download dataset and define transforms
        Args:
            - tasks: List of lists. Each inner list is a description of the
              labels that describe a task
            - samples: Number of samples for each label
            - seed: Random seed
        """
        mean_norm = [0.50]
        std_norm = [0.25]

        dat = ModMNIST
        self.augment_transform = transforms.Compose([
            transforms.RandomCrop(28, padding=4),
            transforms.ToTensor(),
            transforms.Normalize(mean_norm, std_norm),
        ])
        self.vanilla_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean_norm, std=std_norm)])

        # Get dataset
        self.trainset = dat(
            root='./data', train=True, download=True,
            transform=self.augment_transform)
        self.testset = dat(
            root='./data', train=False, download=True,
            transform=self.vanilla_transform)

        # Create a dataset
        self.samples = samples
        tr_ind, te_ind = [], []
        tr_lab, te_lab = [], []

        idx = np.array(range(5000))
        if (seed is not None) and seed >= 0:
            np.random.seed(seed)

        if (seed is None) or (seed >= 0):
            np.random.shuffle(idx)
        else:
            rng = -1 * seed
            idx = np.roll(idx, rng)

        # Filter dataset based on config file
        for task_id, tsk in enumerate(tasks):
            for lab_id, lab in enumerate(tsk):

                task_tr_ind = np.where(np.isin(self.trainset.targets,
                                               [lab]))[0]
                # Consider subset of train dataset and entire test dataset
                task_tr_ind = task_tr_ind[idx[:samples]]
                task_te_ind = np.where(np.isin(self.testset.targets,
                                               [lab]))[0]
                tr_ind.append(task_tr_ind)
                te_ind.append(task_te_ind)
                curlab = (task_id, lab_id)

                tr_vals = [curlab for _ in range(len(task_tr_ind))]
                te_vals = [curlab for _ in range(len(task_te_ind))]

                tr_lab.append(tr_vals)
                te_lab.append(te_vals)

        tr_ind = np.concatenate(tr_ind)
        te_ind = np.concatenate(te_ind)
        self.tr_ind = tr_ind
        self.te_ind = te_ind

        tr_lab, te_lab = np.concatenate(tr_lab), np.concatenate(te_lab)

        self.trainset.data = self.trainset.data[tr_ind]
        self.testset.data = self.testset.data[te_ind]

        self.trainset.targets = [list(it) for it in tr_lab]
        self.testset.targets = [list(it) for it in te_lab]


class RotatedMNISTHandler(MultiTaskDataHandler):
    """
    Rotated MNIST dataset
    """
    def __init__(self,
                 tasks: List[List[int]],
                 samples: int,
                 seed: int = -1) -> None:
        """
        Download dataset and define transforms
        Args:
            - tasks: List of lists. Each inner list is a description of the
              labels that describe a task. If the labe, is 10 * x + y, then the
              label y is rotated by an angle of 10 * x (rotataions are only
              multiples of 10)
            - samples: Number of samples for each label
            - seed: Random seed
        """
        mean_norm = [0.50]
        std_norm = [0.25]

        dat = ModMNIST
        self.augment_transform = transforms.Compose([
            transforms.RandomCrop(28, padding=4),
            transforms.ToTensor(),
            transforms.Normalize(mean_norm, std_norm),
        ])
        self.vanilla_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean_norm, std=std_norm)])

        # Get dataset
        self.trainset = dat(
            root='./data', train=True, download=True,
            transform=self.augment_transform)
        self.testset = dat(
            root='./data', train=False, download=True,
            transform=self.vanilla_transform)

        # Create a dataset
        self.samples = samples
        tr_ind, te_ind = [], []
        tr_lab, te_lab = [], []

        idx = np.array(range(5000))
        if (seed is not None) and seed >= 0:
            np.random.seed(seed)

        if (seed is None) or (seed >= 0):
            np.random.shuffle(idx)
        else:
            rng = -1 * seed
            idx = np.roll(idx, rng)


        for task_id, tsk in enumerate(tasks):
            for lab_id, lab in enumerate(tsk):

                task_tr_ind = np.where(np.isin(self.trainset.targets,
                                               [lab % 10]))[0]
                task_tr_ind = task_tr_ind[idx[:samples]]
                task_te_ind = np.where(np.isin(self.testset.targets,
                                               [lab % 10]))[0]
                tr_ind.append(task_tr_ind)
                te_ind.append(task_te_ind)
                curlab = (task_id, lab_id)

                tr_vals = [curlab for _ in range(len(task_tr_ind))]
                te_vals = [curlab for _ in range(len(task_te_ind))]

                tr_lab.append(tr_vals)
                te_lab.append(te_vals)

        tr_ind = np.concatenate(tr_ind)
        te_ind = np.concatenate(te_ind)
        self.tr_ind = tr_ind
        self.te_ind = te_ind

        tr_lab, te_lab = np.concatenate(tr_lab), np.concatenate(te_lab)
        self.trainset.data = self.trainset.data[tr_ind]
        self.testset.data = self.testset.data[te_ind]

        # Rotate images for each of the tasks based on task_id
        for t_id in range(len(tasks)):
            ang = (tasks[t_id][0] // 10) * 10

            task_tr_flag = tr_lab[:, 0] == t_id
            task_te_flag = te_lab[:, 0] == t_id

            self.trainset.data[task_tr_flag] = transforms.functional.rotate(
                self.trainset.data[task_tr_flag], angle=ang)
            self.testset.data[task_te_flag] = transforms.functional.rotate(
                self.testset.data[task_te_flag], angle=ang)

        self.trainset.targets = [list(it) for it in tr_lab]
        self.testset.targets = [list(it) for it in te_lab]


class PermutedMNISTHandler(MultiTaskDataHandler):
    """
    Initialization for Permuted MNIST dataset
    """
    def __init__(self,
                 tasks: List[List[int]],
                 samples: int,
                 seed: int = -1) -> None:
        """
        Download dataset and define transforms
        Args:
            - tasks: List of lists. Each inner list is a description of the
              labels that describe a task. A label of 10*x + y, then the digit y
              is permuted using random seed (1000*x)
            - samples: Number of samples for each label
            - seed: Random seed
        """
        mean_norm = [0.50]
        std_norm = [0.25]

        dat = ModMNIST
        self.augment_transform = transforms.Compose([
            transforms.RandomCrop(28, padding=4),
            transforms.ToTensor(),
            transforms.Normalize(mean_norm, std_norm),
        ])
        self.vanilla_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean_norm, std=std_norm)])

        # Get dataset
        self.trainset = dat(
            root='./data', train=True, download=True,
            transform=self.augment_transform)
        self.testset = dat(
            root='./data', train=False, download=True,
            transform=self.vanilla_transform)

        # Create a dataset
        self.samples = samples
        tr_ind, te_ind = [], []
        tr_lab, te_lab = [], []

        idx = np.array(range(5000))
        if (seed is not None) and seed >= 0:
            np.random.seed(seed)

        if (seed is None) or (seed >= 0):
            np.random.shuffle(idx)
        else:
            rng = -1 * seed
            idx = np.roll(idx, rng)


        # Filter dataset
        for task_id, tsk in enumerate(tasks):
            for lab_id, lab in enumerate(tsk):

                task_tr_ind = np.where(np.isin(self.trainset.targets,
                                               [lab % 10]))[0]
                task_tr_ind = task_tr_ind[idx[:samples]]
                task_te_ind = np.where(np.isin(self.testset.targets,
                                               [lab % 10]))[0]
                tr_ind.append(task_tr_ind)
                te_ind.append(task_te_ind)
                curlab = (task_id, lab_id)

                tr_vals = [curlab for _ in range(len(task_tr_ind))]
                te_vals = [curlab for _ in range(len(task_te_ind))]

                tr_lab.append(tr_vals)
                te_lab.append(te_vals)

        tr_ind = np.concatenate(tr_ind)
        te_ind = np.concatenate(te_ind)
        self.tr_ind = tr_ind
        self.te_ind = te_ind

        tr_lab, te_lab = np.concatenate(tr_lab), np.concatenate(te_lab)
        self.trainset.data = self.trainset.data[tr_ind]
        self.testset.data = self.testset.data[te_ind]

        # Permute images for each of the tasks based on task_id
        for t_id in range(len(tasks)):
            task_tr_flag = tr_lab[:, 0] == t_id
            task_te_flag = te_lab[:, 0] == t_id

            # Set seed based on task descriptors
            tseed = (tasks[t_id][0] // 10) * 1000
            rng_permute = default_rng(seed=tseed)
            if (tseed == 0):
                idx_permute = np.arange(784)
            else:
                idx_permute = rng_permute.permutation(784)

            self.trainset.data[task_tr_flag] = self.trainset.data[
                task_tr_flag].view(-1, 784)[:, idx_permute].view(-1, 28, 28)
            self.testset.data[task_te_flag] = self.testset.data[
                task_te_flag].view(-1, 784)[:, idx_permute].view(-1, 28, 28)

        self.trainset.targets = [list(it) for it in tr_lab]
        self.testset.targets = [list(it) for it in te_lab]

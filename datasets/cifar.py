import numpy as np
import torchvision
import torchvision.transforms as transforms

from datasets.data import MultiTaskDataHandler
from typing import List


class Cifar100Handler(MultiTaskDataHandler):
    """
    Load CIFAR100 and prepare dataset. Has ability to add noise to some tasks
    by modifying the task configuration supplied through the "tasks" argument in
    __init__
    """
    def __init__(self,
                tasks: List[List[int]],
                samples: int,
                seed: int = -1,
                noise: float = 1.0) -> None:
        """
        Download CIFAR100 and prepare requested config of CIFAR100
        Args:
            - tasks: A List of tasks. Each element in the list is another list
              describing the labels contained in the task. If a 100 is added to
              a task label, then that particular label has randomized labels
            - samples: Number of samples for each label
            - seed: Random seed. A Negative random seed implies that the subset
              of data points chosen is through the np.roll function
        """
        # Cannot use entire data statistics
        # Use this mean/std so that no information leaks from entire dataset
        mean_norm = [0.50, 0.50, 0.50]
        std_norm = [0.25, 0.25, 0.25]

        dat = torchvision.datasets.CIFAR100
        self.augment_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
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

        idx = np.array(range(500))
        if seed >= 0:
            np.random.shuffle(idx)
        else:
            rng = -1 * seed
            idx = np.roll(idx, rng)

        for task_id, tsk in enumerate(tasks):
            for lab_id, lab in enumerate(tsk):

                task_tr_ind = np.where(np.isin(self.trainset.targets,
                                               [lab % 100]))[0]
                task_tr_ind = task_tr_ind[idx[:samples]]
                task_te_ind = np.where(np.isin(self.testset.targets,
                                               [lab % 100]))[0]
                tr_ind.append(task_tr_ind)
                te_ind.append(task_te_ind)
                curlab = (task_id, lab_id)

                if lab < 100:
                    tr_vals = [curlab for _ in range(len(task_tr_ind))]
                else:
                    tr_vals = []
                    for i in range(len(task_tr_ind)):
                        if np.random.uniform(0.0, 1.0) < noise:
                            tr_vals.append((task_id,
                                            np.random.randint(len(tsk))))
                        else:
                            tr_vals.append(curlab)

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


class Cifar10Handler(MultiTaskDataHandler):
    """
    Load CIFAR10 and prepare dataset
    """
    def __init__(self,
                tasks: List[List[int]],
                samples: int,
                seed: int = -1) -> None:
        """
        Download dataset and define transforms
        Args:
            - tasks: A List of tasks. Each element in the list is another list
              describing the labels contained in the task
            - samples: Number of samples for each label
            - seed: Random seed. A Negative random seed implies that the subset
              of data points chosen is through the np.roll function
        """
        # Use this mean/std so that no information leaks from entire dataset
        mean_norm = [0.50, 0.50, 0.50]
        std_norm = [0.25, 0.25, 0.25]

        dat = torchvision.datasets.CIFAR10
        self.augment_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
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
                                               [lab]))[0]
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

#!/usr/bin/env python3
"""
Implementation of the Multihead learner, which is synonymous with
multi-task learning. The setup is identical to the problem outlined by Baxter
(https://arxiv.org/abs/1106.0245).
"""
import argparse
import numpy as np
import torch
import torch.optim as optim
import torch.cuda.amp as amp

from net.build_net import fetch_net
from datasets.build_dataset import fetch_dataclass
from utils.logger import Logger
from utils.config import fetch_configs
from utils.run_net import evaluate, run_epoch


class MultiHead():
    """
    Object for initializing and training a multihead learner
    """
    def __init__(self, args, hp, data_conf):
        """
        Initialize multihead learner

        Params:
          - args:      Arguments from arg parse
          - hp:        JSON config file for hyper-parameters
          - data_conf: JSON config of dataset
        """
        self.args = args
        self.hp = hp

        num_tasks = len(data_conf['tasks'])
        num_classes = len(data_conf['tasks'][0])

        # Random seed
        torch.manual_seed(abs(args.seed))
        np.random.seed(abs(args.seed))

        # Network
        self.net = fetch_net(args, num_tasks, num_classes, hp['dropout'])

        # Get dataset
        dataclass = fetch_dataclass(data_conf['dataset'])
        dataset = dataclass(data_conf['tasks'], args.samples, seed=args.seed)
        self.train_loader = dataset.get_data_loader(hp["batch"], 4, train=True)

        test_loaders = []
        alltrain_loaders = []
        for t_id in range(num_tasks):
            alltrain_loaders.append(
                dataset.get_task_data_loader(t_id, hp['batch'],
                                             4, train=True))
            test_loaders.append(
                dataset.get_task_data_loader(t_id, hp['batch'],
                                             4, train=False))

        # Logger
        self.logger = Logger(test_loaders, alltrain_loaders,
                             num_tasks, num_classes, args, False)

        # Loss and Optimizer
        self.scaler = amp.GradScaler(enabled=args.fp16)
        self.optimizer = optim.SGD(self.net.parameters(), lr=hp['lr'],
                                   momentum=0.9, nesterov=True,
                                   weight_decay=hp['l2_reg'])
        self.lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, args.epochs * len(self.train_loader))

    def train(self, log_interval=5):
        """
        Train the multi-task learner

        Params:
          - log_interval: frequency with which test-set is evaluated
        """
        # Evaluate before start of training:
        train_met = evaluate(self.net, self.train_loader, self.args.gpu)
        self.logger.log_metrics(self.net, train_met, -1)

        # Train multi-head model
        for epoch in range(self.args.epochs):

            train_met = run_epoch(self.net, self.args, self.optimizer,
                                  self.train_loader, self.lr_scheduler,
                                  self.scaler)
            if epoch % log_interval == 0 or epoch == self.args.epochs - 1:
                self.logger.log_metrics(self.net, train_met, epoch)
            else:
                self.logger.log_train(self.net, train_met, epoch)

        return self.net, self.logger.train_accs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int,
                        default=-100,
                        help="Random seed to run")

    # Description of tasks
    # Currently code assumes all tasks have same number of classes
    parser.add_argument("--data_config", type=str,
                        default="./config/dataset/coarse_cifar100.yaml",
                        help="Multi-task config")
    parser.add_argument("--samples", type=int,
                        default=100,
                        help="Number of samples for each label")

    # Hyper-parameters
    parser.add_argument("--hp_config", type=str,
                        default="./config/hyperparam/default.yaml",
                        help="Hyper parameter configuration")

    args = parser.parse_args()
    data_conf = fetch_configs(args.data_config)
    hp_conf = fetch_configs(args.hp_config)
    args.fp16 = args.gpu = torch.cuda.is_available()
    args.model = hp_conf['model']
    args.epochs = hp_conf['epochs']
    args.dataset = data_conf['dataset']

    # Choose best implementation for functions
    # Does sacrifice exact reproducability from random seed
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    model = MultiHead(args, hp_conf, data_conf)
    model.train(log_interval=10)


if __name__ == '__main__':
    main()

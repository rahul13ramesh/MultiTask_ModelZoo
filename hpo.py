#!/usr/bin/env python3
"""
Script for hyperparameter tuning using Ray-tune
"""
import argparse
import random

import numpy as np
import pandas as pd
import nevergrad as ng
import ray
import torch

import torch.optim as optim
import torch.cuda.amp as amp

from ray.tune.suggest.nevergrad import NevergradSearch
from ray.tune.suggest import ConcurrencyLimiter
from ray import tune
from ray.tune.schedulers import ASHAScheduler

from net.build_net import fetch_net
from datasets.build_dataset import fetch_dataclass
from utils.logger import Logger
from utils.config import fetch_configs
from utils.run_net import evaluate, run_epoch


def tune_net(args, dataset_conf):
    """
    Function for tuning the hyper-parameters

    params:
      - args:         Argparse arguments
      - dataset_conf: JSON Dataset configuration

    """
    config = {
        "batch": tune.choice([8, 16, 32, 128]),
        "lr": tune.loguniform(1e-4, 1e-1),
        "l2_reg": tune.loguniform(1e-6, 1e-2),
        "dropout": tune.uniform(0.1, 0.5),
    }

    asha_scheduler = ASHAScheduler(
        time_attr='epoch',
        max_t=args.epochs,
        grace_period=2,
        reduction_factor=3,
        brackets=1)

    algo = NevergradSearch(optimizer=ng.optimizers.OnePlusOne)
    algo = ConcurrencyLimiter(algo, max_concurrent=4)
    ray.init(configure_logging=False)

    result = tune.run(
        tune.with_parameters(train_model, args=args,
                             dataset_conf=dataset_conf, ftune=True,
                             seed=random.randint(0, 1000000)),
        name="multihead",
        resources_per_trial={"cpu": 8, "gpu": 1},
        config=config,
        num_samples=81,
        scheduler=asha_scheduler,
        search_alg=algo,
        checkpoint_at_end=False,
        metric='accuracy',
        mode='max',
        local_dir="./ray_results",
        verbose=1,
        log_to_file=False)

    # Set log-file
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', -1)

    print(vars(args))
    print("Best hyperparameters config: ", result.best_config)
    print("Best hyperparameters name: ", result.best_trial)
    df = result.dataframe(metric="accuracy", mode="max")
    del df['logdir']
    del df['trial_id']
    del df['timesteps_total']
    del df['episodes_total']
    del df['training_iteration']
    del df['experiment_id']
    del df['timestamp']
    del df['pid']
    del df['hostname']
    del df['node_ip']
    del df['time_since_restore']
    del df['timesteps_since_restore']
    del df['iterations_since_restore']
    del df['time_this_iter_s']
    del df['done']
    print(df)
    return result.best_config


def train_model(config, args, dataset_conf, ftune, seed):

    batch_size = config["batch"]
    lr = config["lr"]
    l2_reg = config["l2_reg"]
    dropout = config["dropout"]
    tasks_info = dataset_conf['tasks']

    num_tasks = len(tasks_info)
    num_classes = len(tasks_info[0])

    # Random seed
    torch.manual_seed(abs(seed))
    np.random.seed(abs(seed))

    # Network
    net = fetch_net(args, num_tasks, num_classes, dropout)

    # Get dataset
    dataclass = fetch_dataclass(dataset_conf["dataset"])
    dataset = dataclass(tasks_info, args.samples, seed=seed)
    train_loader = dataset.get_data_loader(batch_size, 4, train=True)

    test_loaders = []
    for t_id in range(num_tasks):
        test_loaders.append(
            dataset.get_task_data_loader(t_id, batch_size, 4, train=False))
    alltrain_loaders = []
    for t_id in range(num_tasks):
        alltrain_loaders.append(
            dataset.get_task_data_loader(t_id, batch_size, 4, train=True))

    # Logger
    logger = Logger(test_loaders, alltrain_loaders,
                    num_tasks, num_classes, args, True)
    # Loss and Optimizer
    scaler = amp.GradScaler(enabled=args.fp16)
    optimizer = optim.SGD(net.parameters(), lr=lr,
                          momentum=0.9, nesterov=True,
                          weight_decay=l2_reg)
    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, args.epochs * len(train_loader))

    # Evaluate before start of training:
    train_met = evaluate(net, train_loader, args.gpu)
    eval_met = logger.log_metrics(net, train_met, -1)
    tune.report(loss=eval_met[1], accuracy=eval_met[0], epoch=0)

    # Tune hyper-parameters
    for epoch in range(args.epochs):
        train_met = run_epoch(net, args, optimizer, train_loader,
                              lr_scheduler, scaler)
        if ftune:
            eval_met = logger.log_metrics(net, train_met, epoch)
            tune.report(loss=eval_met[1], accuracy=eval_met[0], epoch=epoch+1)


def main():
    parser = argparse.ArgumentParser()
    # Description of tasks
    # Currently code assumes all tasks have same number of classes
    parser.add_argument("--data_config", type=str,
                        default="./config/dataset/coarse_cifar100.yaml",
                        help="Multi-task config")
    parser.add_argument("--samples", type=int,
                        default=100,
                        help="Number of samples from each label")

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

    # Tune hyper params
    confg = tune_net(args, data_conf)

    print(confg)


if __name__ == '__main__':
    main()

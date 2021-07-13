import torch
from net.wideresnet import WideResNetMultiTask
from typing import Any


def fetch_net(args: Any,
              num_tasks: int,
              num_cls: int,
              dropout: float = 0.3):
    """
    Create a nearal network to train
    """
    if "mnist" in args.dataset:
        inp_chan = 1
    else:
        inp_chan = 3

    if args.model == "wrn28_10":
        net = WideResNetMultiTask(depth=28, num_task=num_tasks,
                                  num_cls=num_cls, widen_factor=10,
                                  drop_rate=dropout, inp_channels=inp_chan)
    elif args.model == "wrn16_4":
        net = WideResNetMultiTask(depth=16, num_task=num_tasks,
                                  num_cls=num_cls, widen_factor=4,
                                  drop_rate=dropout, inp_channels=inp_chan)
    else:
        raise ValueError("Invalid network")

    if args.gpu:
        net.cuda()
    return net

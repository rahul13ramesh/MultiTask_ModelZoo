# MultitTask_ModelZoo
Implementation of https://arxiv.org/abs/2106.03027

## Setup:

To install a working environment run:
```
conda env create -f env.yaml
```

## Usage

The two key executable files are `modelzoo.py` and `multihead.py`. The `-h`
flag can be used to list the argparse arguments. For example to run Multihead and Model Zoo, execute:

```
python multihead.py --data_config ./config/dataset/coarse_cifar100.yaml \
                    --hp_config ./config/hyperparam/default.yaml        \
                    --samples 100

python modelzoo.py --data_config ./config/dataset/coarse_cifar100.yaml \
                   --hp_config ./config/hyperparam/default.yaml        \
                   --num_rounds 10       \
                   --tasks_per_round 10  \
                   --samples 100
```

To run the continual learning variant of the Model Zoo, add the `--continual` flag. The tasks are presented sequentially with the order prescribed by the data config file.

## Directory Structure

```bash
├── config:                       # Configuration files
│   ├── dataset                    
│   └── hyperparam                  
├── datasets                      # Dataset and Dataloaders
│   ├── build_dataset.py          
│   ├── cifar.py                 
│   ├── data.py                 
│   ├── mnist.py               
│   ├── modmnist.py           
├── hpo.py                        # Hyper-parameter optimization
├── modelzoo.py                   # Implementation of Model Zoo
├── multihead.py                  # Implementation of Multihead
├── net                           # Neural network architectures
│   ├── build_net.py
│   └── wideresnet.py
└── utils                         # Utilities for logging/training
    ├── config.py
    ├── logger.py
    └── run_net.py
```


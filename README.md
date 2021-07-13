# MultitTask_ModelZoo
Implementation of https://arxiv.org/abs/2106.03027

## Setup:

To install a working environment run:
```
conda env create -f env.yaml
```

`hpo.py` is a hyper-paramter optimization script. `modelzoo.py` and
`multihead.py` are implementations of the respective models described
in the paper. The `--continual` flag can be used to run Model Zoo for
continual learning.

## Experiment Environment
- CentOS Linux release 7.9.2009 (Core)
- Slurm Workload Manager
- CUDA 11.8
- Python 3.8
- PyTorch 2.2.2

## Installation
Modify the path in scripts/finetune.sh, scripts/eval.sh, and scripts/train.sh to the following \<PATH> if you run with shell.
```
conda create -p <PATH>/hpe3D python=3.8
conda activate hpe3D
```

`pytorch`

```
pip install torch==2.2.2 torchvision==0.17.2 --index-url https://download.pytorch.org/whl/cu118
```

`pytorch_geometric`

Refer to [official website](https://pytorch-geometric.readthedocs.io/en/2.5.2/notes/installation.html).
2.5.3 version is used in this project.


`sth else`
```
pip install -r requirements.txt

python -m pip install -U scikit-image (if cannot directly install from pip)
```
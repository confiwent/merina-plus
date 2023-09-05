# Artifact appendix for merina_plus
Codes of the paper "MERINA+: Improving Generalization for Neural Video Adaptation via Information-Theoretic Meta-Reinforcement Learning" - Nuowen Kan, et. al. 2023. 

## Abstract
This document illustrates how to obtain the results shown in the paper.

## The environment setup

_Anaconda is suggested to be installed to manage the test environments._

### Prerequisites
- Linux or macOS
- Python >=3.6
- Pytorch >= 1.6.0
- numpy, pandas
- tqdm
- seaborn
- matplotlib
- CPU or NVIDIA GPU + CUDA CuDNN

Install PyTorch. Note that the command of PyTorch intallation depends on the actual compute platform of your own computer, and you can choose appropriate version following the [guide page](https://pytorch.org/get-started/locally/). For example, if you have intalled `CUDA 10.2`, you can intall PyTorch with the latest version by running this Command:

```
conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch
```

__Or__ you can create a specific environment (_many redundant dependencies will be installed_) just via
```
conda env create -f torch.yaml
```

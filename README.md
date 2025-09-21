<h1 align="center"> [IEEE TCSVT'25] MERINA+: Improving Generalization for Neural Video Adaptation via Information-Theoretic Meta-Reinforcement Learning </h1>

<p align="center">
    <a href="https://ieeexplore.ieee.org/abstract/document/11119689"><img src="https://img.shields.io/badge/IEEE-TCSVT-blue.svg" alt="Paper"></a>
    <a href="https://github.com/confiwent/merina-plus"><img src="https://img.shields.io/badge/Github-MERINA+-brightgreen?logo=github" alt="Github"></a>
</p>

by Nuowen Kan, Chenglin Li, Yuankun Jiang, Wenrui Dai, Junni Zou, Hongkai Xiong, and Laura Toni at Shanghai Jiao Tong University

This repository is the official Pytorch implementation of MERINA_Plus

## Abstract
This document illustrates how to obtain the results shown in the paper.

## Code Structure

The repository is organized as follows:

- `main.py`: The main entry point for running the experiments and training the models.
- `algos/`: Contains the core algorithm implementations, including reinforcement learning agents (e.g., PPO), variational autoencoders (VAE), and training scripts for video adaptation.
- `baselines/`: Implements baseline algorithms for comparison, such as MPC, Bola, and other ABR methods.
- `config/`: Configuration files, including argument parsers, data sources, and labels for experiments.
- `envs/`: Environment definitions for video streaming simulations, including video size and VMAF data.
- `utils/`: Utility functions for plotting, data loading, and result visualization.
- `requirements.txt`: Lists the Python dependencies required for the project.
- `torch.yaml`: Conda environment file for easy setup of the PyTorch environment.

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

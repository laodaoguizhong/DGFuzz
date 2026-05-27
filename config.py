import torch
from dataclasses import dataclass

@dataclass
class ExperimentConfig:
    """Paths and runtime environment only; method hyperparameters live in each module."""
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    seed: int = 42
    num_workers: int = 8
    data_root: str = './data'
    output_dir: str = './results'
    num_samples_mnist: int = 1500
    num_samples_cifar10: int = 1500

config = ExperimentConfig()


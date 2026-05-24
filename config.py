import torch
from dataclasses import dataclass
from typing import Tuple

@dataclass
class ExperimentConfig:
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    seed: int = 42
    num_workers: int = 8
    data_root: str = './data'
    num_samples_mnist: int = 1500
    num_samples_cifar10: int = 1500
    pgd_eps: float = 0.03
    pgd_steps: int = 40
    pgd_alpha: float = 0.01
    fab_eps: float = 0.03
    fab_steps: int = 20
    fab_restarts: int = 1
    mm_eps: float = 0.03
    mm_steps: int = 20
    mm_alpha: float = 0.01
    fmn_eps: float = 0.03
    fmn_steps: int = 20
    fmn_alpha: float = 0.01
    fmn_gamma: float = 0.05
    umap_n_neighbors: int = 15
    umap_min_dist: float = 0.1
    rotation_range: Tuple[float, float] = (-15, 15)
    translation_range: Tuple[int, int] = (-4, 4)
    scale_range: Tuple[float, float] = (0.9, 1.1)
    brightness_range: Tuple[float, float] = (0.8, 1.2)
    contrast_range: Tuple[float, float] = (0.8, 1.2)
    gaussian_noise_range: Tuple[float, float] = (0.01, 0.05)
    gaussian_blur_sigma: Tuple[float, float] = (0.5, 1.5)
    dgfuzz_bandwidth_mnist: float = 0.3
    dgfuzz_bandwidth_cifar10: float = 0.5
    dgfuzz_tau_r: float = 0.1
    dgfuzz_top_b_ratio: float = 0.5
    dgfuzz_psi1: float = 0.8
    dgfuzz_psi2: float = 2.0
    dgfuzz_cosine_threshold_mnist: float = 0.1
    dgfuzz_cosine_threshold_cifar10: float = 0.15
    output_dir: str = './results'

config = ExperimentConfig()


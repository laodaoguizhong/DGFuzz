import torch
from dataclasses import dataclass
from typing import List, Tuple

@dataclass
class ExperimentConfig:
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    seed: int = 42
    num_workers: int = 8
    datasets: List[str] = ('mnist', 'cifar10')
    data_root: str = './data'
    mnist_models: List[str] = ('lenet1', 'lenet4', 'lenet5')
    cifar10_models: List[str] = ('resnet20', 'resnet50')
    num_samples_mnist: int = 1500
    num_samples_cifar10: int = 1500
    attack_methods: List[str] = ('fgsm', 'pgd', 'deepfool', 'autoattack')
    batch_size_train: int = 256
    batch_size_inference: int = 512
    batch_size_attack: int = 128
    fgsm_eps: List[float] = (0.03, 0.05)
    pgd_eps: float = 0.03
    pgd_steps: int = 40
    pgd_alpha: float = 0.01
    autoattack_enabled: bool = True
    autoattack_eps: float = 0.03
    autoattack_num_samples: int = 1500
    n_clusters_list: List[int] = (5, 10, 15)
    tsne_perplexity: int = 50
    tsne_n_iter: int = 1000
    tsne_n_jobs: int = 8
    umap_n_neighbors: int = 15
    umap_min_dist: float = 0.1
    num_seeds_per_cluster: int = 50
    num_mutations_per_seed: int = 10
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
    checkpoint_dir: str = './checkpoints'
    figure_dir: str = './figures'
    use_amp: bool = True
    pin_memory: bool = True
    non_blocking: bool = True
config = ExperimentConfig()

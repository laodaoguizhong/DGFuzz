import os
import sys
from pathlib import Path
from typing import Dict, Any
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
try:
    from .adv import AdversarialSampleGenerator
except ImportError:
    _PROJECT_ROOT = Path(__file__).resolve().parents[1]
    if str(_PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(_PROJECT_ROOT))
    from utils.adv import AdversarialSampleGenerator
from models import get_model
from config import config as exp_config

def _load_model(dataset: str, model: str, device: torch.device):
    model_obj = get_model(model, dataset)
    ckpt = f'./checkpoints/{dataset}_{model}.pth'
    ckpt_data = torch.load(ckpt, map_location=device)
    if isinstance(ckpt_data, dict) and 'model_state_dict' in ckpt_data:
        model_obj.load_state_dict(ckpt_data['model_state_dict'])
    else:
        model_obj.load_state_dict(ckpt_data)
    return model_obj.to(device).eval()

def _load_dataset(dataset: str) -> DataLoader:
    if dataset == 'mnist':
        ds = datasets.MNIST(exp_config.data_root, train=False, download=True, transform=transforms.ToTensor())
    else:
        ds = datasets.CIFAR10(exp_config.data_root, train=False, download=True, transform=transforms.ToTensor())
    return DataLoader(ds, batch_size=128, shuffle=False, num_workers=exp_config.num_workers)

def load_advlib(model: torch.nn.Module, dataset: str, model_name: str, cfg: Dict[str, Any], device: torch.device):
    exp_config.device = str(device)
    num_samples = cfg.get('num_adversarial_samples')
    if num_samples is None:
        num_samples = exp_config.num_samples_mnist if dataset == 'mnist' else exp_config.num_samples_cifar10
    path = os.path.join(exp_config.output_dir, dataset, model_name, 'adversarial_library.npz')
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if os.path.exists(path) and (not cfg.get('force_regenerate', False)):
        print(f'Loading cached adversarial library from: {path}')
        return dict(np.load(path, allow_pickle=True))
    print(f'Generating adversarial library for {dataset}-{model_name} ...')
    loader = _load_dataset(dataset)
    gen = AdversarialSampleGenerator(model, exp_config, dataset)
    lib = gen.build_adversarial_library(loader, num_samples=num_samples)
    np.savez(path, **lib)
    print(f'Adversarial library saved to: {path}')
    return lib

def run_advlib(cfg: Dict[str, Any]):
    dataset = cfg['dataset']
    model_name = cfg['model']
    device = torch.device(cfg['device'])
    print(f'===== Adv-library Generator on {dataset}-{model_name} ({device}) =====')
    model = _load_model(dataset, model_name, device)
    lib = load_advlib(model, dataset, model_name, cfg, device)
    print(f"Done: adversarial_library_size={int(len(lib.get('samples', [])))}")
    return lib
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Adv-library Generator')
    parser.add_argument('--dataset', type=str, default=None, choices=['mnist', 'cifar10'], help='Dataset name')
    parser.add_argument('--model', type=str, default=None, help='Model name (e.g., lenet1, lenet4, lenet5, resnet20, resnet50)')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to use')
    parser.add_argument('--num_adversarial_samples', type=int, default=None, help='Number of adversarial samples to generate (per dataset-model)')
    parser.add_argument('--force_regenerate', action='store_true', help='Force regenerate adversarial library even if cached')
    args = vars(parser.parse_args())
    model_map = {'mnist': ['lenet1', 'lenet4', 'lenet5'], 'cifar10': ['resnet20', 'resnet50', 'resnet18', 'resnet34', 'densenet121', 'googlenet', 'mobilenet_v2', 'vgg13_bn', 'vgg16_bn']}
    if args['dataset'] and args['model']:
        run_advlib(args)
    else:
        for ds, models in model_map.items():
            for m in models:
                cfg = args.copy()
                cfg['dataset'] = ds
                cfg['model'] = m
                run_advlib(cfg)

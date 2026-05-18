import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import random
import math
import os
import argparse
from .resnet import resnet18, resnet34, resnet50
from .densenet import densenet121
from .googlenet import googlenet
from .mobilenetv2 import mobilenet_v2
from .vgg import vgg11_bn, vgg13_bn, vgg16_bn, vgg19_bn

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class Cutout:

    def __init__(self, n_holes=1, length=16):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        h, w = (img.size(1), img.size(2))
        mask = np.ones((h, w), np.float32)
        for _ in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)
            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)
            mask[y1:y2, x1:x2] = 0.0
        mask = torch.from_numpy(mask).expand_as(img)
        return img * mask

def mixup_data(x, y, alpha=0.2):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = (y, y[index])
    return (mixed_x, y_a, y_b, lam)

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

def get_lr(epoch, warmup_epochs, total_epochs, base_lr):
    if epoch < warmup_epochs:
        return base_lr * (epoch + 1) / warmup_epochs
    else:
        progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
        return base_lr * 0.5 * (1 + math.cos(math.pi * progress))

def adjust_learning_rate(optimizer, epoch, config):
    lr = get_lr(epoch, config.warmup_epochs, config.epochs, config.lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

def get_dataloaders(dataset='cifar10', batch_size=256, num_workers=4, cutout=True, cutout_length=16):
    if dataset.lower() == 'cifar10':
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.247, 0.2435, 0.2616)
        train_transform_list = [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(), transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1), transforms.ToTensor(), transforms.Normalize(mean, std)]
        if cutout:
            train_transform_list.append(Cutout(n_holes=1, length=cutout_length))
        train_transform = transforms.Compose(train_transform_list)
        test_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
        train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)
        test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)
    elif dataset.lower() == 'mnist':
        mean = (0.1307,)
        std = (0.3081,)
        train_transform = transforms.Compose([transforms.RandomRotation(10), transforms.ToTensor(), transforms.Normalize(mean, std)])
        test_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
        train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=train_transform)
        test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=test_transform)
    else:
        raise ValueError(f'Unsupported dataset: {dataset}')
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return (train_loader, test_loader)

def get_model(model_name, dataset='cifar10', pretrained=False, device='cuda'):
    model_name = model_name.lower()
    if model_name == 'resnet18':
        model = resnet18(pretrained=pretrained, device=device)
    elif model_name == 'resnet34':
        model = resnet34(pretrained=pretrained, device=device)
    elif model_name == 'resnet50':
        model = resnet50(pretrained=pretrained, device=device)
    elif model_name == 'densenet121':
        model = densenet121(pretrained=pretrained, device=device)
    elif model_name == 'googlenet':
        model = googlenet(pretrained=pretrained, device=device)
    elif model_name == 'mobilenet_v2':
        model = mobilenet_v2(pretrained=pretrained, device=device)
    elif model_name == 'vgg11_bn':
        model = vgg11_bn(pretrained=pretrained, device=device)
    elif model_name == 'vgg13_bn':
        model = vgg13_bn(pretrained=pretrained, device=device)
    elif model_name == 'vgg16_bn':
        model = vgg16_bn(pretrained=pretrained, device=device)
    elif model_name == 'vgg19_bn':
        model = vgg19_bn(pretrained=pretrained, device=device)
    else:
        raise ValueError(f'Unsupported model: {model_name}. Only support: resnet18/34/50, densenet121, googlenet, mobilenet_v2, vgg11/13/16/19_bn')
    return model

def train_one_epoch(model, train_loader, criterion, optimizer, config, epoch):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = (inputs.to(config.device), targets.to(config.device))
        if config.mixup:
            inputs, targets_a, targets_b, lam = mixup_data(inputs, targets, config.mixup_alpha)
        optimizer.zero_grad()
        outputs = model(inputs)
        if config.mixup:
            loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
        else:
            loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        if config.mixup:
            correct += (lam * predicted.eq(targets_a).sum().float() + (1 - lam) * predicted.eq(targets_b).sum().float()).item()
        else:
            correct += predicted.eq(targets).sum().item()
    return (total_loss / len(train_loader), 100.0 * correct / total)

@torch.no_grad()
def evaluate(model, test_loader, criterion, config):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    for inputs, targets in test_loader:
        inputs, targets = (inputs.to(config.device), targets.to(config.device))
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    return (total_loss / len(test_loader), 100.0 * correct / total)

class Config:

    def __init__(self):
        self.batch_size = 256
        self.num_workers = 4
        self.epochs = 300
        self.warmup_epochs = 5
        self.lr = 0.1
        self.momentum = 0.9
        self.weight_decay = 0.0005
        self.nesterov = True
        self.cutout = True
        self.cutout_length = 16
        self.mixup = True
        self.mixup_alpha = 0.2
        self.label_smoothing = 0.1
        self.data_root = './data'
        self.checkpoint_dir = './checkpoints'
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main(model_name, dataset='cifar10', pretrained=False):
    print('=' * 70)
    print(f'Training {model_name.upper()} on {dataset.upper()}')
    print('=' * 70)
    set_seed(42)
    config = Config()
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    train_loader, test_loader = get_dataloaders(dataset=dataset, batch_size=config.batch_size, num_workers=config.num_workers, cutout=config.cutout, cutout_length=config.cutout_length)
    print(f'Dataset: {dataset}')
    print(f'   Train samples: {len(train_loader.dataset)}')
    print(f'   Test samples: {len(test_loader.dataset)}')
    model = get_model(model_name, dataset, pretrained, config.device).to(config.device)
    print(f'Model: {model_name}')
    print(f'   Parameters: {sum((p.numel() for p in model.parameters())):,}')
    print(f'   Device: {config.device}')
    criterion = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)
    optimizer = optim.SGD(model.parameters(), lr=config.lr, momentum=config.momentum, weight_decay=config.weight_decay, nesterov=config.nesterov)
    best_acc = 0
    best_epoch = 0
    print('\n' + '=' * 70)
    print('Starting training...')
    print('=' * 70)
    print(f'Epochs: {config.epochs} | Warmup: {config.warmup_epochs}')
    print(f'Batch Size: {config.batch_size} | LR: {config.lr}')
    print(f'Cutout: {config.cutout} | Mixup: {config.mixup}')
    print(f'Label Smoothing: {config.label_smoothing}')
    print('=' * 70 + '\n')
    for epoch in range(config.epochs):
        current_lr = adjust_learning_rate(optimizer, epoch, config)
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, config, epoch)
        test_loss, test_acc = evaluate(model, test_loader, criterion, config)
        print(f'Epoch [{epoch + 1:3d}/{config.epochs}] LR: {current_lr:.6f} | Train Loss: {train_loss:.4f}, Acc: {train_acc:.2f}% | Test Loss: {test_loss:.4f}, Acc: {test_acc:.2f}%', end='')
        if test_acc > best_acc:
            best_acc = test_acc
            best_epoch = epoch + 1
            print(' New best!')
        else:
            print()
    save_path = os.path.join(config.checkpoint_dir, f'{model_name}_{dataset}.pth')
    torch.save(model.state_dict(), save_path)
    print('\n' + '=' * 70)
    print('Training complete!')
    print(f'Best test accuracy: {best_acc:.2f}% (epoch {best_epoch})')
    print(f'Model saved to: {save_path}')
    print('=' * 70)
    return best_acc
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Unified Training Script for CIFAR-10')
    parser.add_argument('--model', type=str, required=True, choices=['resnet18', 'resnet34', 'densenet121', 'googlenet', 'mobilenet_v2', 'vgg13_bn', 'vgg16_bn'], help='Model name')
    parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10'], help='Dataset name (only cifar10 supported)')
    parser.add_argument('--pretrained', action='store_true', help='Use pretrained weights')
    args = parser.parse_args()
    main(args.model, args.dataset, args.pretrained)

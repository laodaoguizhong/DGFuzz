import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from torchattacks import PGD, FAB
import warnings
import time
warnings.filterwarnings('ignore')

# VRM attack defaults (fixed in implementation).
ATTACK_EPS = 0.03
ATTACK_ALPHA = 0.01
PGD_STEPS = 40
FAB_STEPS = 20
FAB_RESTARTS = 1
MM_STEPS = 20
FMN_STEPS = 20
FMN_GAMMA = 0.05

class AdversarialSampleGenerator:

    def __init__(self, model, device: str, dataset_name: str):
        self.device = device
        self.model = model.to(device).eval()
        self.dataset_name = dataset_name
        is_mnist = dataset_name == 'mnist'
        self.batch_sizes = {'fab': 1024 if is_mnist else 128, 'pgd': 2048 if is_mnist else 256, 'mm': 2048 if is_mnist else 256, 'fmn': 2048 if is_mnist else 256}
        torch.backends.cudnn.benchmark = True
        print(f"\n{'=' * 60}")
        print(f'Adversarial Generator | {dataset_name.upper()} | {self.device}')
        print(f"{'=' * 60}\n")

    def _predict_np(self, x_tensor):
        with torch.no_grad():
            logits = self.model(x_tensor)
            return logits.argmax(dim=1).detach().cpu().numpy()

    def _mm_attack(self, x, y):
        eps, alpha, steps = ATTACK_EPS, ATTACK_ALPHA, MM_STEPS
        x_orig = x.detach()
        x_adv = x_orig.clone().detach()
        x_adv = x_adv + (torch.rand_like(x_adv) * 2 - 1) * eps
        x_adv = torch.clamp(x_adv, 0.0, 1.0)
        for _ in range(steps):
            x_adv.requires_grad_(True)
            logits = self.model(x_adv)
            true_logits = logits.gather(1, y.unsqueeze(1)).squeeze(1)
            mask = torch.zeros_like(logits, dtype=torch.bool)
            mask.scatter_(1, y.unsqueeze(1), True)
            other_logits = logits.masked_fill(mask, float('-inf')).max(dim=1).values
            margin = true_logits - other_logits
            loss = margin.mean()
            grad = torch.autograd.grad(loss, x_adv)[0]
            x_adv = x_adv.detach() - alpha * grad.sign()
            delta = torch.clamp(x_adv - x_orig, min=-eps, max=eps)
            x_adv = torch.clamp(x_orig + delta, 0.0, 1.0).detach()
        return x_adv

    def _fmn_attack(self, x, y):
        max_eps, steps, alpha, gamma = ATTACK_EPS, FMN_STEPS, ATTACK_ALPHA, FMN_GAMMA
        x_orig = x.detach()
        x_adv = x_orig.clone().detach()
        eps_t = torch.full((x.shape[0], 1, 1, 1), max_eps, device=x.device)
        for _ in range(steps):
            x_adv.requires_grad_(True)
            logits = self.model(x_adv)
            loss = nn.CrossEntropyLoss()(logits, y)
            grad = torch.autograd.grad(loss, x_adv)[0]
            x_tmp = x_adv.detach() + alpha * grad.sign()
            delta = x_tmp - x_orig
            delta = torch.max(torch.min(delta, eps_t), -eps_t)
            x_adv = torch.clamp(x_orig + delta, 0.0, 1.0).detach()
            preds = self._predict_np(x_adv)
            success = torch.from_numpy((preds != y.detach().cpu().numpy()).astype(np.float32)).to(x.device)
            success = success.view(-1, 1, 1, 1)
            eps_t = torch.where(success > 0.5, torch.clamp(eps_t * (1.0 - gamma), min=0.0001, max=max_eps), torch.clamp(eps_t * (1.0 + gamma), min=0.0001, max=max_eps))
        return x_adv

    def _run_attack(self, name, x, y, attack_fn, batch_size):
        results = {'samples': [], 'indices': [], 'labels': [], 'adv_labels': [], 'success': 0, 'failed': 0}
        for start in tqdm(range(0, len(x), batch_size), desc=f'  {name}', ncols=80, leave=False):
            end = min(start + batch_size, len(x))
            x_batch, y_batch = (x[start:end], y[start:end])
            try:
                x_tensor = torch.from_numpy(x_batch).float().to(self.device)
                y_tensor = torch.from_numpy(y_batch).long().to(self.device)
                x_adv_tensor = attack_fn(x_tensor, y_tensor)
                x_adv = x_adv_tensor.detach().cpu().numpy()
                preds = self._predict_np(x_adv_tensor)
                success_mask = preds != y_batch
                for i in np.where(success_mask)[0]:
                    results['samples'].append(x_adv[i])
                    results['indices'].append(start + i)
                    results['labels'].append(y_batch[i])
                    results['adv_labels'].append(preds[i])
                results['success'] += success_mask.sum()
                results['failed'] += (~success_mask).sum()
            except Exception:
                results['failed'] += len(x_batch)
        return results

    def build_adversarial_library(self, data_loader, num_samples):
        all_samples, all_labels = ([], [])
        for images, labels in data_loader:
            all_samples.append(images.numpy())
            all_labels.append(labels.numpy())
            if sum((s.shape[0] for s in all_samples)) >= num_samples:
                break
        all_samples = np.concatenate(all_samples)[:num_samples]
        all_labels = np.concatenate(all_labels)[:num_samples]
        print(f'Loaded {len(all_samples)} samples\n')
        adv_library = {'samples': [], 'original_indices': [], 'attack_methods': [], 'original_labels': [], 'adv_labels': []}
        stats = {}
        pgd_attack = PGD(self.model, eps=ATTACK_EPS, alpha=ATTACK_ALPHA, steps=PGD_STEPS, random_start=True)
        fab_attack = FAB(self.model, eps=ATTACK_EPS, steps=FAB_STEPS, n_restarts=FAB_RESTARTS)
        attacks = [('PGD', [('pgd', lambda x, y: pgd_attack(x, y))]), ('FAB', [('fab', lambda x, y: fab_attack(x, y))]), ('MM Attack', [('mm', self._mm_attack)]), ('FMN', [('fmn', self._fmn_attack)])]
        for phase, attack_info in enumerate(attacks, 1):
            phase_name, attack_list = (attack_info[0], attack_info[1])
            max_samples = attack_info[2] if len(attack_info) > 2 else len(all_samples)
            print(f"{'=' * 60}\nPhase {phase}/4: {phase_name}")
            print(f"{'=' * 60}")
            start_time = time.time()
            phase_success, phase_failed = (0, 0)
            x_phase = all_samples[:max_samples]
            y_phase = all_labels[:max_samples]
            for method_name, attack_fn in attack_list:
                bs = self.batch_sizes.get(method_name.split('_')[0], self.batch_sizes['pgd'])
                res = self._run_attack(method_name, x_phase, y_phase, attack_fn, bs)
                for i, sample in enumerate(res['samples']):
                    adv_library['samples'].append(sample)
                    adv_library['original_indices'].append(res['indices'][i])
                    adv_library['attack_methods'].append(method_name)
                    adv_library['original_labels'].append(res['labels'][i])
                    adv_library['adv_labels'].append(res['adv_labels'][i])
                phase_success += res['success']
                phase_failed += res['failed']
            elapsed = time.time() - start_time
            stats[phase_name.lower()] = {'success': phase_success, 'failed': phase_failed, 'time': elapsed}
            print(f'{phase_name}: {phase_success} success, {phase_failed} failed ({elapsed:.1f}s)\n')
        for key in adv_library:
            adv_library[key] = np.array(adv_library[key])
        print(f"{'=' * 60}")
        print(f"Total: {len(adv_library['samples'])} adversarial samples")
        if len(adv_library['samples']) > 0:
            methods, counts = np.unique(adv_library['attack_methods'], return_counts=True)
            for m, c in zip(methods, counts):
                print(f"   {m:20s}: {c:5d} ({c / len(adv_library['samples']) * 100:.1f}%)")
        print(f"{'=' * 60}\n")
        return adv_library

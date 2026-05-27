import numpy as np
import torch
import json
import os
import time
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict
import matplotlib.pyplot as plt
from tqdm import tqdm
from utils.mutate import MutationEngine
from utils.cluster import ClusteringAnalyzer
from utils.strategy import FuzzerStrategy, DGFuzzStrategy
from utils.runtime import DEFECT_SIGMA

@dataclass
class DGFuzzConfig:
    max_samples: int = 100000
    max_time_hours: float = 24.0
    early_stop_rounds: int = 1000
    mutations_per_seed: int = 10
    mutation_strategies: List[str] = None
    max_strategy_combo: int = 3
    confidence_threshold: float = DEFECT_SIGMA
    cosine_distance_threshold: float = 0.05
    semantic_cosine_threshold: float = 0.1
    max_seed_pool_size: int = 5000
    dedup_cosine_threshold: float = 0.03
    scheduler: str = 'round_robin'

    def __post_init__(self):
        if self.mutation_strategies is None:
            self.mutation_strategies = ['rotate', 'translate', 'scale', 'brightness', 'contrast', 'noise', 'blur']

def cosine_distance(v1: np.ndarray, v2: np.ndarray) -> float:
    v1_norm = v1 / (np.linalg.norm(v1) + 1e-08)
    v2_norm = v2 / (np.linalg.norm(v2) + 1e-08)
    return 1 - np.dot(v1_norm, v2_norm)

def _pred_and_confidence(logits: np.ndarray) -> Tuple[int, float]:
    lr = logits.astype(np.float64)
    pred = int(lr.argmax())
    e = np.exp(lr - lr.max())
    probs = e / e.sum()
    return pred, float(probs.max())

class DefectPattern:

    def __init__(self, sample: np.ndarray, true_label: int, pred_label: int, confidence: float, feature: np.ndarray, parent_seed_id: Optional[int]=None):
        self.sample = sample
        self.true_label = true_label
        self.pred_label = pred_label
        self.confidence = confidence
        self.feature = feature
        self.parent_seed_id = parent_seed_id
        self.timestamp = time.time()
        self.is_misclassification = pred_label != true_label
        self.is_low_confidence = confidence < 0.3
        self.error_pattern = f'{true_label}->{pred_label}'

class SeedPool:

    def __init__(self, config: DGFuzzConfig, analyzer: ClusteringAnalyzer):
        self.config = config
        self.analyzer = analyzer
        self.seeds: List[Dict] = []
        self.seed_features = []
        self.current_index = 0

    def add_seed(self, sample: np.ndarray, label: int, feature: Optional[np.ndarray]=None, priority: float=1.0, generation: int=0):
        if feature is None:
            feature = self.analyzer.extract_features(sample.reshape(1, *sample.shape))[0]
        if self._is_duplicate_cosine(feature):
            return False
        if len(self.seeds) >= self.config.max_seed_pool_size:
            self._remove_lowest_priority()
        self.seeds.append({'sample': sample, 'label': label, 'feature': feature, 'priority': priority, 'generation': generation, 'mutation_count': 0})
        self.seed_features.append(feature)
        return True

    def _is_duplicate_cosine(self, feature: np.ndarray) -> bool:
        if len(self.seed_features) == 0:
            return False
        distances = [cosine_distance(feature, sf) for sf in self.seed_features]
        return np.min(distances) < self.config.dedup_cosine_threshold

    def _remove_lowest_priority(self):
        if not self.seeds:
            return
        priorities = [s['priority'] for s in self.seeds]
        min_idx = np.argmin(priorities)
        self.seeds.pop(min_idx)
        self.seed_features.pop(min_idx)

    def get_next_seed(self) -> Optional[Dict]:
        if not self.seeds:
            return None
        if self.config.scheduler == 'round_robin':
            seed = self.seeds[self.current_index]
            self.current_index = (self.current_index + 1) % len(self.seeds)
            seed['mutation_count'] += 1
            return seed
        elif self.config.scheduler == 'priority_based':
            priorities = np.array([s['priority'] for s in self.seeds])
            probs = priorities / priorities.sum()
            idx = np.random.choice(len(self.seeds), p=probs)
            self.seeds[idx]['mutation_count'] += 1
            return self.seeds[idx]

    def __len__(self):
        return len(self.seeds)

class DGFuzzFuzzer:

    def __init__(self, model: torch.nn.Module, config: DGFuzzConfig, analyzer: ClusteringAnalyzer, mutation_engine: MutationEngine, device: torch.device, strategy: Optional[FuzzerStrategy]=None):
        self.model = model.to(device).eval()
        self.config = config
        self.analyzer = analyzer
        self.mutation_engine = mutation_engine
        self.device = device
        self.strategy = strategy
        self.use_strategy = strategy is not None
        self.seed_pool = SeedPool(config, analyzer)
        self.stats = {'total_samples_tested': 0, 'total_defects_found': 0, 'unique_defects': 0, 'duplicate_defects': 0, 'start_time': None, 'first_defect_time': None, 'defects_by_type': {'misclassification': 0, 'low_confidence': 0}, 'defects_by_error_pattern': defaultdict(int), 'generation_stats': {}, 'time_series': []}
        self.defects: List[DefectPattern] = []
        self.defect_features = []
        self.error_pattern_index = defaultdict(list)
        self.rounds_without_new_defect = 0

    def _detect_defect(self, sample: np.ndarray, true_label: int, parent_seed: Dict) -> Optional[DefectPattern]:
        with torch.no_grad():
            input_tensor = torch.FloatTensor(sample).unsqueeze(0).to(self.device)
            logits_np, feat_np = self.analyzer.forward_logits_and_features(input_tensor)
            pred_label, confidence = _pred_and_confidence(logits_np[0])
        feature = feat_np[0]
        is_misclass = pred_label != true_label
        if is_misclass:
            defect = DefectPattern(sample=sample, true_label=true_label, pred_label=pred_label, confidence=confidence, feature=feature, parent_seed_id=id(parent_seed))
            return defect
        return None

    def _is_unique_defect(self, defect: DefectPattern) -> bool:
        if len(self.defect_features) == 0:
            return True
        same_pattern_defects = self.error_pattern_index.get(defect.error_pattern, [])
        if same_pattern_defects:
            for existing_idx in same_pattern_defects:
                existing_defect = self.defects[existing_idx]
                cos_dist = cosine_distance(defect.feature, existing_defect.feature)
                if cos_dist < self.config.semantic_cosine_threshold:
                    return False
        distances = [cosine_distance(defect.feature, df) for df in self.defect_features]
        min_distance = np.min(distances)
        return min_distance > self.config.cosine_distance_threshold

    def _update_statistics(self, defect: DefectPattern, parent_generation: int):
        self.stats['total_defects_found'] += 1
        if self._is_unique_defect(defect):
            self.stats['unique_defects'] += 1
            defect_idx = len(self.defects)
            self.defects.append(defect)
            self.defect_features.append(defect.feature)
            self.error_pattern_index[defect.error_pattern].append(defect_idx)
            if self.stats['first_defect_time'] is None:
                self.stats['first_defect_time'] = time.time()
            if defect.is_misclassification:
                self.stats['defects_by_type']['misclassification'] += 1
            if defect.is_low_confidence:
                self.stats['defects_by_type']['low_confidence'] += 1
            self.stats['defects_by_error_pattern'][defect.error_pattern] += 1
            gen_key = f'gen_{parent_generation}'
            self.stats['generation_stats'][gen_key] = self.stats['generation_stats'].get(gen_key, 0) + 1
            elapsed = time.time() - self.stats['start_time']
            self.stats['time_series'].append((elapsed, self.stats['unique_defects']))
            self.rounds_without_new_defect = 0
        else:
            self.stats['duplicate_defects'] += 1
            self.rounds_without_new_defect += 1

    def run_fuzzing(self) -> Dict:
        print('\n' + '=' * 80)
        print('Phase 2: Saturation Fuzzing')
        print('=' * 80)
        if self.use_strategy:
            print(f'[Strategy] {type(self.strategy).__name__}')
            return self._run_fuzzing_with_strategy()
        else:
            return self._run_fuzzing_legacy()

    def _run_fuzzing_with_strategy(self) -> Dict:
        self.stats['start_time'] = time.time()
        max_time_seconds = self.config.max_time_hours * 3600
        pbar = tqdm(total=self.config.max_samples, desc='Fuzzing Progress', ncols=100)
        try:
            while self.stats['total_samples_tested'] < self.config.max_samples:
                elapsed = time.time() - self.stats['start_time']
                if elapsed > max_time_seconds:
                    print(f'\n[Time Limit] Reached time limit ({self.config.max_time_hours}h)')
                    break
                if self.rounds_without_new_defect >= self.config.early_stop_rounds:
                    print(f'\n[Early Stop] No new defects for {self.config.early_stop_rounds} rounds')
                    break
                seed = self.strategy.select_seed(self.seed_pool)
                if seed is None:
                    print('\n[Error] Seed pool empty, stopping')
                    break
                for _ in range(self.config.mutations_per_seed):
                    mutated = self.strategy.mutate(seed)
                    self.stats['total_samples_tested'] += 1
                    with torch.no_grad():
                        input_tensor = torch.FloatTensor(mutated).unsqueeze(0).to(self.device)
                        l2_feature = None
                        if self.analyzer is not None:
                            logits, l2_feat_np = self.analyzer.forward_logits_and_features(input_tensor)
                            logits = logits[0]
                            l2_feature = l2_feat_np[0].astype(np.float32, copy=False)
                        else:
                            output = self.model(input_tensor)
                            logits = output.cpu().numpy()[0]
                        pred_label, confidence = _pred_and_confidence(logits)
                    metadata = {'pred_label': pred_label, 'confidence': confidence, 'logits': logits, 'parent_seed': seed}
                    if l2_feature is not None:
                        metadata['l2_feature'] = l2_feature
                    is_interesting, defect_info = self.strategy.is_interesting(mutated, seed['label'], metadata)
                    if is_interesting:
                        if defect_info is not None:
                            pred_label = defect_info.get('pred_label')
                            true_label = defect_info.get('true_label')
                            confidence = defect_info.get('confidence', 0.0)
                            if l2_feature is not None:
                                feature = l2_feature
                            else:
                                feature = self.analyzer.extract_features(mutated.reshape(1, *mutated.shape))[0]
                            defect = DefectPattern(sample=mutated, true_label=true_label, pred_label=pred_label, confidence=confidence, feature=feature, parent_seed_id=id(seed))
                            self._update_statistics(defect, seed.get('generation', 0))
                        feature_for_seed = None
                        if self.analyzer is not None:
                            feature_for_seed = l2_feature if l2_feature is not None else self.analyzer.extract_features(mutated.reshape(1, *mutated.shape))[0]
                        priority_val = 2.0
                        self.seed_pool.add_seed(sample=mutated, label=seed['label'], feature=feature_for_seed, priority=priority_val, generation=seed.get('generation', 0) + 1)
                        self.rounds_without_new_defect = 0
                    pbar.update(1)
                    if elapsed > 0:
                        rate = self.stats['unique_defects'] / elapsed * 3600
                    else:
                        rate = 0.0
                    postfix = {'Unique': self.stats['unique_defects'], 'Dup': self.stats.get('duplicate_defects', 0), 'Rate': f'{rate:.1f}/h', 'Seeds': len(self.seed_pool)}
                    pbar.set_postfix(postfix)
                    if torch.cuda.is_available():
                        try:
                            del input_tensor
                        except Exception:
                            pass
                        torch.cuda.empty_cache()
        finally:
            pbar.close()
        self.stats['total_time'] = time.time() - self.stats['start_time']
        return self._generate_report()

    def _run_fuzzing_legacy(self) -> Dict:
        self.stats['start_time'] = time.time()
        max_time_seconds = self.config.max_time_hours * 3600
        pbar = tqdm(total=self.config.max_samples, desc='Fuzzing Progress', ncols=100)
        try:
            while self.stats['total_samples_tested'] < self.config.max_samples:
                elapsed = time.time() - self.stats['start_time']
                if elapsed > max_time_seconds:
                    print(f'\n[Time Limit] Reached time limit ({self.config.max_time_hours}h)')
                    break
                if self.rounds_without_new_defect >= self.config.early_stop_rounds:
                    print(f'\n[Early Stop] No new defects for {self.config.early_stop_rounds} rounds')
                    break
                seed = self.seed_pool.get_next_seed()
                if seed is None:
                    print('\n[Error] Seed pool empty, stopping')
                    break
                me = self.mutation_engine
                ops = {'rotate': me.rotate, 'translate': me.translate, 'scale': me.scale, 'brightness': me.adjust_brightness, 'contrast': me.adjust_contrast, 'noise': me.add_gaussian_noise, 'blur': me.gaussian_blur}
                strategies = np.random.choice(self.config.mutation_strategies, size=np.random.randint(1, self.config.max_strategy_combo + 1), replace=False)
                for _ in range(self.config.mutations_per_seed):
                    mutated = seed['sample'].copy()
                    for strategy in strategies:
                        mutated = ops[strategy](mutated)
                    self.stats['total_samples_tested'] += 1
                    defect = self._detect_defect(mutated, seed['label'], seed)
                    if defect is not None:
                        self._update_statistics(defect, seed['generation'])
                        self.seed_pool.add_seed(sample=mutated, label=seed['label'], feature=defect.feature, priority=2.0, generation=seed['generation'] + 1)
                    pbar.update(1)
                    if elapsed > 0:
                        rate = self.stats['unique_defects'] / elapsed * 3600
                    else:
                        rate = 0.0
                    pbar.set_postfix({'Unique': self.stats['unique_defects'], 'Dup': self.stats['duplicate_defects'], 'Rate': f'{rate:.1f}/h', 'Seeds': len(self.seed_pool)})
        finally:
            pbar.close()
        self.stats['total_time'] = time.time() - self.stats['start_time']
        return self._generate_report()

    def _generate_report(self) -> Dict:
        report = {'config': {'max_samples': self.config.max_samples, 'max_time_hours': self.config.max_time_hours, 'mutations_per_seed': self.config.mutations_per_seed, 'cosine_distance_threshold': self.config.cosine_distance_threshold, 'semantic_cosine_threshold': self.config.semantic_cosine_threshold}, 'summary': {'total_samples_tested': self.stats['total_samples_tested'], 'total_defects_found': self.stats['total_defects_found'], 'unique_defects': self.stats['unique_defects'], 'duplicate_defects': self.stats['duplicate_defects'], 'deduplication_rate': self.stats['duplicate_defects'] / max(self.stats['total_defects_found'], 1), 'total_time_hours': self.stats['total_time'] / 3600, 'defect_discovery_rate': self.stats['unique_defects'] / (self.stats['total_time'] / 3600), 'first_defect_time': self.stats['first_defect_time'] - self.stats['start_time'] if self.stats['first_defect_time'] else None}, 'defects_by_type': self.stats['defects_by_type'], 'defects_by_error_pattern': dict(self.stats['defects_by_error_pattern']), 'generation_stats': self.stats['generation_stats'], 'time_series': self.stats['time_series']}
        return report

def release_cuda_memory() -> None:
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

def run_dgfuzz_experiment(config: Dict) -> Dict:
    try:
        return _run_dgfuzz_experiment_impl(config)
    finally:
        release_cuda_memory()

def _run_dgfuzz_experiment_impl(config: Dict) -> Dict:
    from models import get_model
    from torchvision import datasets, transforms
    dataset_name = config['dataset']
    model_name = config['model']
    device = torch.device(config['device'])
    print(f"\n{'=' * 80}")
    print(f'Experiment: {dataset_name.upper()}-{model_name.upper()}')
    print(f"{'=' * 80}")
    model = get_model(model_name, dataset_name)
    checkpoint = torch.load(f'./checkpoints/{dataset_name}_{model_name}.pth', map_location=device)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model = model.to(device).eval()
    dataset_class = datasets.MNIST if dataset_name == 'mnist' else datasets.CIFAR10
    test_dataset = dataset_class(root='./data', train=False, download=True, transform=transforms.ToTensor())
    from config import config as paths_config
    from utils.cluster import hdbscan_params_for
    from utils.runtime import dgfuzz_defaults_for_dataset
    adv_library_path = f'./results/{dataset_name}/{model_name}/adversarial_library.npz'
    if not os.path.exists(adv_library_path):
        raise FileNotFoundError(f'Adversarial library not found: {adv_library_path}')
    adv_library = dict(np.load(adv_library_path, allow_pickle=True))
    hdbscan_mcs, hdbscan_ms = hdbscan_params_for(dataset_name, model_name)
    analyzer = ClusteringAnalyzer(model, str(device), seed=paths_config.seed, hdbscan_min_cluster_size=hdbscan_mcs, hdbscan_min_samples=hdbscan_ms)
    features = analyzer.extract_features(adv_library['samples'])
    runtime_library_path = f'./results/{dataset_name}/{model_name}/adversarial_library_runtime.npz'
    np.savez(runtime_library_path, **adv_library, features=features)
    cluster_results = analyzer.perform_hdbscan_clustering_highdim(features)
    mutation_engine = MutationEngine()
    dg_defaults = dgfuzz_defaults_for_dataset(dataset_name)

    def _cfg(name: str, default):
        value = config.get(name, None)
        return default if value is None else value

    def _to_bool(v, default=True):
        if v is None:
            return default
        if isinstance(v, bool):
            return v
        if isinstance(v, (int, float)):
            return bool(v)
        if isinstance(v, str):
            return v.strip().lower() in ('1', 'true', 'yes', 'y', 'on')
        return bool(v)
    dgfuzz_config = DGFuzzConfig()
    for _k in ('max_samples', 'max_time_hours', 'early_stop_rounds', 'mutations_per_seed'):
        if config.get(_k) is not None:
            setattr(dgfuzz_config, _k, config[_k])
    dgfuzz_config.confidence_threshold = _cfg('dgfuzz_sigma', dg_defaults['sigma'])
    strategy = DGFuzzStrategy(model=model, mutation_engine=mutation_engine, device=device, config=dgfuzz_config, dataset=dataset_name, adv_library_path=runtime_library_path, analyzer=analyzer, cluster_labels=cluster_results['labels'], bandwidth=_cfg('dgfuzz_bandwidth', dg_defaults['bandwidth']), tau_r=_cfg('dgfuzz_tau_r', dg_defaults['tau_r']), top_b_ratio=_cfg('dgfuzz_top_b_ratio', dg_defaults['top_b_ratio']), psi1=_cfg('dgfuzz_psi1', dg_defaults['psi1']), psi2=_cfg('dgfuzz_psi2', dg_defaults['psi2']), cosine_threshold=_cfg('dgfuzz_cosine_threshold', dg_defaults['cosine_threshold']), tau_h=_cfg('dgfuzz_tau_h', dg_defaults['tau_h']), enable_vrm=True, enable_seed_init=_cfg('enable_seed_init', True), enable_adaptive_schedule=_cfg('enable_adaptive_schedule', True))
    fuzzer = DGFuzzFuzzer(model=model, config=dgfuzz_config, analyzer=analyzer, mutation_engine=mutation_engine, device=device, strategy=strategy)
    n_seeds = strategy.initialize_seeds(seed_pool=fuzzer.seed_pool, test_dataset=test_dataset, extra_info={'dataset_name': dataset_name, 'model_name': model_name})
    if n_seeds == 0:
        raise RuntimeError('Seed pool initialization failed: no valid seeds')
    results = fuzzer.run_fuzzing()
    exp_tag = 'dgfuzz'
    enable_seed_init = _to_bool(config.get('enable_seed_init', True), True)
    enable_adaptive_schedule = _to_bool(config.get('enable_adaptive_schedule', True), True)
    is_full_dgfuzz = enable_seed_init and enable_adaptive_schedule
    ablation_name = config.get('ablation_name', None)
    if not ablation_name:
        if not enable_seed_init:
            ablation_name = 'seed_init'
        elif not enable_adaptive_schedule:
            ablation_name = 'adaptive'
    if is_full_dgfuzz:
        save_dir = f'./results/{dataset_name}/{model_name}/{exp_tag}'
        os.makedirs(save_dir, exist_ok=True)
        result_path = os.path.join(save_dir, f'{exp_tag}_results.json')
    else:
        ablation_name = str(ablation_name or 'ablation').strip()
        save_dir = './results/ablation'
        os.makedirs(save_dir, exist_ok=True)
        result_path = os.path.join(save_dir, f'{model_name}_{ablation_name}.json')
    with open(result_path, 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    _visualize_results(results, save_dir, model_name)
    print(f"\n{'=' * 80}")
    print(f'[OK] Fuzzing finished (strategy={exp_tag})')
    print(f"Unique defects: {results['summary']['unique_defects']}")
    print(f"Duplicate defects: {results['summary']['duplicate_defects']}")
    print(f"Deduplication rate: {results['summary']['deduplication_rate'] * 100:.1f}%")
    print(f"Total time: {results['summary']['total_time_hours']:.2f}h")
    print(f"Discovery rate: {results['summary']['defect_discovery_rate']:.2f} defects/h")
    print(f'Results saved to: {result_path}')
    print(f"{'=' * 80}\n")
    return results

def _visualize_results(results: Dict, save_dir: str, model_name: str):
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    ax1 = axes[0, 0]
    if results['time_series']:
        times, defects = zip(*results['time_series'])
        times = np.array(times) / 3600
        ax1.plot(times, defects, 'b-', linewidth=2, label='Unique Defects')
        ax1.fill_between(times, defects, alpha=0.3)
        ax1.set_xlabel('Time (hours)', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Cumulative Unique Defects', fontsize=12, fontweight='bold')
        ax1.set_title(f'{model_name.upper()} - Defect Discovery Over Time', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend(fontsize=11)
    ax2 = axes[0, 1]
    if results['generation_stats']:
        generations = sorted(results['generation_stats'].keys(), key=lambda x: int(x.split('_')[1]))
        counts = [results['generation_stats'][g] for g in generations]
        gen_labels = [g.replace('gen_', 'Gen ') for g in generations]
        ax2.bar(range(len(generations)), counts, color='steelblue', alpha=0.7)
        ax2.set_xticks(range(len(generations)))
        ax2.set_xticklabels(gen_labels, rotation=45)
        ax2.set_xlabel('Seed Generation', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Defects Discovered', fontsize=12, fontweight='bold')
        ax2.set_title(f'{model_name.upper()} - Defects by Generation', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')
    ax3 = axes[1, 0]
    if results['defects_by_error_pattern']:
        sorted_patterns = sorted(results['defects_by_error_pattern'].items(), key=lambda x: x[1], reverse=True)[:10]
        patterns, counts = zip(*sorted_patterns)
        ax3.barh(range(len(patterns)), counts, color='coral', alpha=0.7)
        ax3.set_yticks(range(len(patterns)))
        ax3.set_yticklabels(patterns, fontsize=10)
        ax3.set_xlabel('Count', fontsize=12, fontweight='bold')
        ax3.set_title(f'{model_name.upper()} - Top 10 Error Patterns', fontsize=14, fontweight='bold')
        ax3.grid(True, alpha=0.3, axis='x')
        ax3.invert_yaxis()
    ax4 = axes[1, 1]
    labels = ['Unique\nDefects', 'Duplicate\nDefects']
    sizes = [results['summary']['unique_defects'], results['summary']['duplicate_defects']]
    colors = ['#2ecc71', '#e74c3c']
    explode = (0.1, 0)
    ax4.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', shadow=True, startangle=90, textprops={'fontsize': 12, 'fontweight': 'bold'})
    ax4.set_title(f'{model_name.upper()} - Deduplication Effect', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'fuzz_comprehensive_analysis3.png'), dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='DGFuzz fuzzing')
    parser.add_argument('--dataset', type=str, default=None, choices=['mnist', 'cifar10'], help='Dataset name (if not specified, run all)')
    parser.add_argument('--model', type=str, default=None, help='Model name (if not specified, run all)')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--max_samples', type=int, default=15000)
    parser.add_argument('--max_time_hours', type=float, default=1.0)
    parser.add_argument('--early_stop_rounds', type=int, default=500)
    parser.add_argument('--mutations_per_seed', type=int, default=10)
    parser.add_argument('--cosine_distance_threshold', type=float, default=0.05)
    parser.add_argument('--semantic_cosine_threshold', type=float, default=0.1)
    parser.add_argument('--dgfuzz_bandwidth', type=float, default=None)
    parser.add_argument('--dgfuzz_tau_r', type=float, default=None)
    parser.add_argument('--dgfuzz_top_b_ratio', type=float, default=None)
    parser.add_argument('--dgfuzz_psi1', type=float, default=None)
    parser.add_argument('--dgfuzz_psi2', type=float, default=None)
    parser.add_argument('--dgfuzz_cosine_threshold', type=float, default=None)
    parser.add_argument('--dgfuzz_tau_h', type=float, default=None, help='Normalized entropy threshold for fallback exploration (region V)')
    parser.add_argument('--dgfuzz_sigma', type=float, default=None, help='Minimum confidence on misclassified class (Def. 1)')
    parser.add_argument('--enable_seed_init', type=int, choices=[0, 1], default=1)
    parser.add_argument('--enable_adaptive_schedule', type=int, choices=[0, 1], default=1)
    parser.add_argument('--ablation_name', type=str, default=None, help='Ablation run name when a switch is off; saved under results/ablation/')
    parser.add_argument('--strategy', type=str, default='dgfuzz', choices=['dgfuzz'], help='Strategy name (only dgfuzz is supported)')
    args = parser.parse_args()
    args.enable_seed_init = bool(args.enable_seed_init)
    args.enable_adaptive_schedule = bool(args.enable_adaptive_schedule)
    model_map = {'mnist': ['lenet1', 'lenet4', 'lenet5'], 'cifar10': ['resnet20', 'resnet50', 'resnet18', 'resnet34', 'densenet121', 'googlenet', 'mobilenet_v2', 'vgg13_bn', 'vgg16_bn']}
    if args.dataset and args.model:
        config = vars(args).copy()
        config['dataset'] = args.dataset
        config['model'] = args.model
        try:
            run_dgfuzz_experiment(config)
        except Exception as e:
            print(f'[Error] {args.dataset}-{args.model}: {e}')
    else:
        base_config = vars(args).copy()
        for ds, models in model_map.items():
            for m in models:
                config = base_config.copy()
                config['dataset'] = ds
                config['model'] = m
                try:
                    run_dgfuzz_experiment(config)
                except Exception as e:
                    print(f'[Error] {ds}-{m}: {e}')
                    continue

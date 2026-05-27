import numpy as np
import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional, Any
from utils.runtime import build_runtime, dgfuzz_defaults_for_dataset, DEFECT_SIGMA

class FuzzerStrategy(ABC):

    def __init__(self, model: nn.Module, mutation_engine, device: torch.device, config):
        self.model = model.to(device).eval()
        self.mutation_engine = mutation_engine
        self.device = device
        self.config = config
        self.stats = {'total_samples_tested': 0, 'total_defects_found': 0, 'unique_defects': 0}

    @abstractmethod
    def select_seed(self, seed_pool) -> Optional[Dict]:
        pass

    @abstractmethod
    def mutate(self, seed: Dict) -> np.ndarray:
        pass

    @abstractmethod
    def is_interesting(self, sample: np.ndarray, true_label: int, metadata: Dict) -> Tuple[bool, Optional[Dict]]:
        pass

    def reset_stats(self):
        self.stats = {'total_samples_tested': 0, 'total_defects_found': 0, 'unique_defects': 0}

    def initialize_seeds(self, seed_pool, test_dataset, extra_info: Optional[Dict]=None) -> int:
        return len(seed_pool)

def create_runtime(dataset: str, adv_library_path: str, cluster_labels, cluster_centers=None, **kwargs):
    kwargs.pop('train_dataloader', None)
    kwargs.pop('analyzer', None)
    defaults = dgfuzz_defaults_for_dataset(dataset)
    return build_runtime(adv_library_path=adv_library_path, cluster_labels=cluster_labels, cluster_centers=cluster_centers, bandwidth=kwargs.pop('bandwidth', defaults['bandwidth']), tau_r=kwargs.pop('tau_r', defaults['tau_r']), top_b_ratio=kwargs.pop('top_b_ratio', defaults['top_b_ratio']), psi1=kwargs.pop('psi1', defaults['psi1']), psi2=kwargs.pop('psi2', defaults['psi2']), cosine_threshold=kwargs.pop('cosine_threshold', defaults['cosine_threshold']), tau_h=kwargs.pop('tau_h', defaults['tau_h']), enable_vrm=kwargs.pop('enable_vrm', True), enable_adaptive_schedule=kwargs.pop('enable_adaptive_schedule', True), **kwargs)

class DGFuzzStrategy(FuzzerStrategy):

    def __init__(self, model: nn.Module, mutation_engine, device: torch.device, config, dataset: str, adv_library_path: str, cluster_labels, cluster_centers=None, analyzer=None, **kwargs):
        super().__init__(model, mutation_engine, device, config)
        self.analyzer = analyzer
        self._dataset = dataset
        self._adv_library_path = adv_library_path
        self._cluster_original_indices = set()
        self._enable_seed_init = bool(kwargs.pop('enable_seed_init', True))
        try:
            _data = np.load(self._adv_library_path, allow_pickle=True)
            if 'original_indices' in _data and cluster_labels is not None:
                original_indices = np.asarray(_data['original_indices'])
                labels = np.asarray(cluster_labels)
                if len(original_indices) == len(labels):
                    valid_mask = labels != -1
                    in_cluster_original = original_indices[valid_mask]
                    self._cluster_original_indices = set((int(i) for i in in_cluster_original.tolist()))
                else:
                    self._cluster_original_indices = set((int(i) for i in original_indices.tolist()))
        except Exception:
            self._cluster_original_indices = set()
        kwargs.pop('train_dataloader', None)
        self._runtime = create_runtime(dataset=dataset, adv_library_path=adv_library_path, cluster_labels=cluster_labels, cluster_centers=cluster_centers, **kwargs)

    def _extract_feature(self, sample: np.ndarray) -> np.ndarray:
        if self.analyzer is not None:
            return self.analyzer.extract_features(sample.reshape(1, *sample.shape))[0]
        return sample.reshape(-1).astype(np.float32)

    def initialize_seeds(self, seed_pool, test_dataset, extra_info: Optional[Dict]=None) -> int:
        if not self._enable_seed_init:
            n_total = len(test_dataset)
            target_seeds = 1000
            rng = np.random.default_rng()
            picked = rng.permutation(np.arange(n_total))
            selected_correct = 0
            added = 0
            for i in picked:
                x, y = test_dataset[int(i)]
                sample = x.numpy()
                with torch.no_grad():
                    input_tensor = torch.FloatTensor(sample).unsqueeze(0).to(self.device)
                    logits = self.model(input_tensor)
                    pred = int(torch.argmax(logits, dim=1).item())
                if pred != int(y):
                    continue
                selected_correct += 1
                feature = self._extract_feature(sample)
                ok = seed_pool.add_seed(sample=sample, label=int(y), feature=feature, priority=1.0, generation=0)
                added += int(bool(ok))
                if selected_correct >= target_seeds:
                    break
            return added
        regular_seeds: List[Dict[str, Any]] = []
        regular_labels: List[int] = []
        direct_seeds: List[Dict[str, Any]] = []
        for i in range(len(test_dataset)):
            x, y = test_dataset[i]
            sample = x.numpy()
            s = {'sample': sample, 'label': int(y), 'priority': 1.0, 'generation': 0}
            if i in self._cluster_original_indices:
                direct_seeds.append(s)
            else:
                regular_seeds.append(s)
                regular_labels.append(int(y))
        max_regular = 2000
        if len(regular_seeds) > max_regular:
            labels_arr = np.asarray(regular_labels, dtype=np.int64)
            unique_labels = np.unique(labels_arr)
            rng = np.random.default_rng(42)
            per_class = max_regular // max(len(unique_labels), 1)
            picked_idx: List[int] = []
            picked_set = set()
            for c in unique_labels:
                cls_idx = np.where(labels_arr == c)[0]
                take = min(per_class, len(cls_idx))
                if take > 0:
                    chosen = rng.choice(cls_idx, size=take, replace=False)
                    chosen_list = chosen.tolist()
                    picked_idx.extend(chosen_list)
                    picked_set.update(chosen_list)
            remain_pool = [int(i) for i in range(len(regular_seeds)) if i not in picked_set]
            remain_need = max_regular - len(picked_idx)
            if remain_need > 0 and remain_pool:
                extra = rng.choice(np.asarray(remain_pool), size=min(remain_need, len(remain_pool)), replace=False)
                picked_idx.extend(extra.tolist())
            picked_idx = sorted(set(picked_idx))[:max_regular]
            regular_seeds = [regular_seeds[i] for i in picked_idx]
            regular_labels = [regular_labels[i] for i in picked_idx]

        def feature_extractor(batch):
            if isinstance(batch, dict):
                batch = batch.get('sample')
            arr = np.asarray(batch)
            if arr.ndim == 3:
                arr = arr[None, ...]
            if self.analyzer is not None:
                return self.analyzer.extract_features(arr)
            return arr.reshape(arr.shape[0], -1).astype(np.float32)

        def logit_extractor(batch):
            if isinstance(batch, dict):
                batch = batch.get('sample')
            arr = np.asarray(batch, dtype=np.float32)
            if arr.ndim == 3:
                arr = arr[None, ...]
            x = torch.from_numpy(arr).to(self.device)
            with torch.no_grad():
                out = self.model(x)
            return out.detach().cpu().numpy()
        infer_batch = 256 if self._dataset == 'mnist' else 32
        regular_logits_list: List[np.ndarray] = []
        regular_features_list: List[np.ndarray] = []
        if regular_seeds:
            regular_samples = np.stack([np.asarray(s['sample'], dtype=np.float32) for s in regular_seeds], axis=0)
            for i in range(0, regular_samples.shape[0], infer_batch):
                batch = regular_samples[i:i + infer_batch]
                logits_batch = logit_extractor(batch)
                feats_batch = feature_extractor(batch)
                regular_logits_list.append(logits_batch.astype(np.float32))
                regular_features_list.append(np.asarray(feats_batch, dtype=np.float32))
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            regular_logits = np.concatenate(regular_logits_list, axis=0)
            regular_features = np.concatenate(regular_features_list, axis=0)
        else:
            regular_logits = np.empty((0, 0), dtype=np.float32)
            regular_features = None
        keep_count = None
        print(f'[DGFuzz Seed Init] regular_candidates={len(regular_seeds)}, direct_in_cluster={len(direct_seeds)}, regular_keep_target={keep_count or 0}')
        regular_kept = self._runtime.initialize_seeds(regular_seeds, feature_extractor, regular_logits, regular_labels, keep_count=keep_count, features_arr=regular_features, feature_batch_size=32)
        if direct_seeds:
            direct_samples = np.stack([np.asarray(s['sample'], dtype=np.float32) for s in direct_seeds], axis=0)
            direct_features_list: List[np.ndarray] = []
            for i in range(0, direct_samples.shape[0], infer_batch):
                batch = direct_samples[i:i + infer_batch]
                feats_batch = feature_extractor(batch)
                direct_features_list.append(np.asarray(feats_batch, dtype=np.float32))
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            direct_features = np.concatenate(direct_features_list, axis=0)
        else:
            direct_features = np.empty((0, 0), dtype=np.float32)
        direct_added = self._runtime.append_direct_seeds(direct_seeds, direct_features)
        qv = len(self._runtime._fallback_pool)
        print(f'[DGFuzz Seed Init] regular_kept={regular_kept}, direct_added={direct_added}, Q_V={qv}, total_initialized={regular_kept + direct_added}')
        return regular_kept + direct_added

    def select_seed(self, seed_pool=None):
        return self._runtime.select_seed(seed_pool)

    def mutate(self, seed: Dict) -> np.ndarray:
        original = seed['sample'].copy()
        me = self.mutation_engine
        ops = {'rotate': me.rotate, 'translate': me.translate, 'scale': me.scale, 'brightness': me.adjust_brightness, 'contrast': me.adjust_contrast, 'noise': me.add_gaussian_noise, 'blur': me.gaussian_blur}
        max_n = min(4, len(self.config.mutation_strategies) + 1)
        for _ in range(5):
            mutated = original.copy()
            for strategy in np.random.choice(self.config.mutation_strategies, size=np.random.randint(1, max_n), replace=False):
                mutated = ops[strategy](mutated)
            if me.check_semantic_preservation(original, mutated):
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                return mutated
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return mutated

    def is_interesting(self, sample, true_label: int, metadata: Dict):
        pred_label = metadata.get('pred_label')
        confidence = metadata.get('confidence', 0.0)
        pre_f = metadata.get('l2_feature')
        if pre_f is not None:
            feature_vec = np.asarray(pre_f, dtype=np.float32).reshape(-1)
        else:
            feature_vec = self._extract_feature(sample)
        is_unique = False
        if pred_label is not None:
            is_unique = bool(self._runtime.is_interesting(feature_vec, int(true_label), int(pred_label)))
        is_defect = False
        defect_info = None
        sigma = getattr(self.config, 'confidence_threshold', DEFECT_SIGMA)
        if pred_label is not None and int(pred_label) != int(true_label) and float(confidence) >= float(sigma):
            is_defect = True
            defect_info = {'pred_label': int(pred_label), 'true_label': int(true_label), 'confidence': float(confidence), 'type': 'misclassification'}
        parent_seed = metadata.get('parent_seed')
        parent_generation = int(parent_seed.get('generation', 0)) if isinstance(parent_seed, dict) else 0
        parent_priority = float(parent_seed.get('priority', 1.0)) if isinstance(parent_seed, dict) else 1.0
        runtime_seed = {'sample': sample, 'label': int(true_label), 'priority': parent_priority, 'generation': parent_generation + 1}
        self._runtime.update_after_trial(seed=runtime_seed, feature_vec=feature_vec, y_true=int(true_label), y_pred=int(pred_label) if pred_label is not None else int(true_label), is_defect=is_defect, is_unique=is_unique)
        del feature_vec, runtime_seed
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return (is_unique or is_defect, defect_info)

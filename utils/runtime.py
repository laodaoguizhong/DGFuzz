from __future__ import annotations
import heapq
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
try:
    import torch
except ImportError:
    torch = None
logger = logging.getLogger(__name__)
_EPS = 1e-08

# DGFuzz method defaults (paper); override via fuzz.py CLI kwargs only.
TAU_R = 0.1
TOP_B_RATIO = 0.5
PSI1 = 0.8
PSI2 = 2.0
TAU_H = 0.95
DEFECT_SIGMA = 0.3
BANDWIDTH_BY_DATASET = {'mnist': 0.3, 'cifar10': 0.5}
COSINE_THRESHOLD_BY_DATASET = {'mnist': 0.1, 'cifar10': 0.15}


def dgfuzz_defaults_for_dataset(dataset: str) -> Dict[str, float]:
    ds = dataset.lower()
    return {
        'tau_r': TAU_R,
        'top_b_ratio': TOP_B_RATIO,
        'psi1': PSI1,
        'psi2': PSI2,
        'tau_h': TAU_H,
        'sigma': DEFECT_SIGMA,
        'bandwidth': BANDWIDTH_BY_DATASET.get(ds, BANDWIDTH_BY_DATASET['cifar10']),
        'cosine_threshold': COSINE_THRESHOLD_BY_DATASET.get(ds, COSINE_THRESHOLD_BY_DATASET['cifar10']),
    }

@dataclass(order=True)
class _PriorityItem:
    neg_priority: float
    seed: Any = field(compare=False)

@dataclass
class ClusterState:
    cluster_id: int
    center: np.ndarray
    size: int
    unique_defects: int = 0
    total_trials: int = 0
    priority: float = 0.0

    @property
    def defect_rate(self) -> float:
        if self.total_trials == 0:
            return 0.0
        return self.unique_defects / self.total_trials

class UniqueDefectChecker:

    def __init__(self, cosine_threshold: float=COSINE_THRESHOLD_BY_DATASET['cifar10']):
        self.threshold = cosine_threshold
        self._store: Dict[Tuple[int, int], List[np.ndarray]] = defaultdict(list)

    def is_unique(self, feature_vec: np.ndarray, y_true: int, y_pred: int) -> bool:
        key = (int(y_true), int(y_pred))
        fv = self._normalize(feature_vec)
        bucket = self._store[key]
        for stored in bucket:
            cos_sim = float(np.dot(fv, stored))
            cos_dist = 1.0 - cos_sim
            if cos_dist <= self.threshold:
                return False
        bucket.append(fv)
        return True

    @staticmethod
    def _normalize(v: np.ndarray) -> np.ndarray:
        v = v.astype(np.float32).ravel()
        norm = np.linalg.norm(v)
        return v / (norm + _EPS)

    @property
    def total_unique(self) -> int:
        return sum((len(v) for v in self._store.values()))

def _gaussian_kernel(x: np.ndarray, center: np.ndarray, bandwidth: float=BANDWIDTH_BY_DATASET['cifar10']) -> float:
    diff = x.astype(np.float32).ravel() - center.astype(np.float32).ravel()
    sq_dist = float(np.dot(diff, diff))
    return float(np.exp(-sq_dist / (2.0 * bandwidth ** 2 + _EPS)))

def compute_region_strength(feature_vec: np.ndarray, clusters: List[ClusterState], bandwidth: float=BANDWIDTH_BY_DATASET['cifar10']) -> Tuple[int, float]:
    best_id = -1
    best_r = -1.0
    for cs in clusters:
        r = _gaussian_kernel(feature_vec, cs.center, bandwidth)
        if r > best_r:
            best_r = r
            best_id = cs.cluster_id
    return (best_id, best_r)

class SeedInitializer:

    def __init__(self, clusters: List[ClusterState], bandwidth: float=BANDWIDTH_BY_DATASET['cifar10'], tau_r: float=TAU_R, top_b_ratio: float=TOP_B_RATIO, enable_vrm: bool=True):
        self.clusters = clusters
        self.bandwidth = bandwidth
        self.tau_r = tau_r
        self.top_b_ratio = top_b_ratio
        self.enable_vrm = enable_vrm

    @staticmethod
    def _stack_seed_samples(seeds: List[Any]) -> np.ndarray:
        chunks: List[np.ndarray] = []
        for s in seeds:
            if isinstance(s, dict) and 'sample' in s:
                chunks.append(np.asarray(s['sample'], dtype=np.float32))
            else:
                chunks.append(np.asarray(s, dtype=np.float32))
        return np.stack(chunks, axis=0)

    def run(self, seeds: List[Any], feature_extractor, logits_arr: np.ndarray, true_labels: List[int], keep_count: Optional[int]=None, features_arr: Optional[np.ndarray]=None, feature_batch_size: int=32) -> Tuple[List[Tuple[Any, int, float]], List[Tuple[Any, int, float]]]:
        if not seeds:
            return [], []
        logger.info('[SeedInit] Stage 1: computing boundary sensitivity ...')
        if features_arr is not None:
            features = np.asarray(features_arr, dtype=np.float32)
            if features.ndim != 2 or features.shape[0] != len(seeds):
                raise ValueError(f'features_arr must be (N, D), N={len(seeds)}, got {features.shape}')
        else:
            row_list: List[np.ndarray] = []
            for start in range(0, len(seeds), feature_batch_size):
                chunk = seeds[start:start + feature_batch_size]
                stacked = self._stack_seed_samples(chunk)
                feats = np.asarray(feature_extractor(stacked), dtype=np.float32)
                if feats.ndim == 1:
                    feats = feats.reshape(1, -1)
                for r in range(feats.shape[0]):
                    row_list.append(feats[r].ravel())
                if torch is not None and torch.cuda.is_available():
                    torch.cuda.empty_cache()
            features = np.stack(row_list, axis=0)
        logits_arr = np.asarray(logits_arr, dtype=np.float32)
        margins = self._compute_margins(logits_arr, true_labels)
        b_scores = 1.0 / (margins + _EPS)
        n = len(seeds)
        b_thresh = np.quantile(b_scores, 1.0 - self.top_b_ratio)
        stage1_mask = b_scores >= b_thresh
        candidate_idx = np.where(stage1_mask)[0]
        logger.info('[SeedInit] Stage 1 filter: %d / %d passed (theta_B=%.4f)', len(candidate_idx), n, b_thresh)
        if len(candidate_idx) == 0:
            logger.warning('[SeedInit] Stage 1: no candidates; relaxing threshold, keeping all samples.')
            candidate_idx = np.arange(n)
        primary: List[Tuple[Any, int, float]] = []
        fallback: List[Tuple[Any, int, float]] = []
        default_cluster_id = self.clusters[0].cluster_id if self.clusters else -1
        for idx in candidate_idx:
            fv = features[idx]
            if self.enable_vrm:
                best_id, best_r = compute_region_strength(fv, self.clusters, self.bandwidth)
            else:
                best_id, best_r = (default_cluster_id, 1.0)
            best_r = max(best_r, _EPS)
            item = (seeds[idx], best_id, best_r)
            if not self.enable_vrm or best_r >= self.tau_r:
                primary.append(item)
            else:
                fallback.append(item)
        logger.info('[SeedInit] Stage 2: Q=%d, Q_V=%d / %d (tau_R=%.4f)', len(primary), len(fallback), len(candidate_idx), self.tau_r)
        if keep_count is not None and keep_count > 0 and len(primary) > keep_count:
            primary.sort(key=lambda x: x[2], reverse=True)
            primary = primary[:keep_count]
            logger.info('[SeedInit] Regular pool size cap: kept Q=%d.', len(primary))
        return primary, fallback

    @staticmethod
    def _compute_margins(logits: np.ndarray, true_labels: List[int]) -> np.ndarray:
        n, _ = logits.shape
        margins = np.zeros(n, dtype=np.float32)
        for i, y in enumerate(true_labels):
            lt = logits[i, y]
            others = np.concatenate([logits[i, :y], logits[i, y + 1:]])
            margins[i] = lt - others.max()
        return margins

class PrioritySeedQueue:

    def __init__(self):
        self._heap: List[_PriorityItem] = []

    def push(self, seed: Any, priority: float) -> None:
        heapq.heappush(self._heap, _PriorityItem(-priority, seed))

    def pop(self) -> Optional[Any]:
        if not self._heap:
            return None
        item = heapq.heappop(self._heap)
        return item.seed

    def pop_with_priority(self) -> Optional[Tuple[Any, float]]:
        if not self._heap:
            return None
        item = heapq.heappop(self._heap)
        return (item.seed, float(-item.neg_priority))

    def __len__(self) -> int:
        return len(self._heap)

    def is_empty(self) -> bool:
        return len(self._heap) == 0

class DGFuzzRuntimeCore:

    def __init__(self, clusters: List[ClusterState], bandwidth: float=BANDWIDTH_BY_DATASET['cifar10'], tau_r: float=TAU_R, psi1: float=PSI1, psi2: float=PSI2, cosine_threshold: float=COSINE_THRESHOLD_BY_DATASET['cifar10'], top_b_ratio: float=TOP_B_RATIO, tau_h: float=TAU_H, enable_vrm: bool=True, enable_adaptive_schedule: bool=True):
        self.clusters = clusters
        self.bandwidth = bandwidth
        self.tau_r = tau_r
        self.tau_h = tau_h
        self.psi1 = psi1
        self.psi2 = psi2
        self.top_b_ratio = top_b_ratio
        self.enable_vrm = enable_vrm
        self.enable_adaptive_schedule = enable_adaptive_schedule
        self._defect_checker = UniqueDefectChecker(cosine_threshold)
        self._seed_queue = PrioritySeedQueue()
        total_size = sum((cs.size for cs in clusters)) or 1
        for cs in clusters:
            cs.priority = cs.size / total_size
        self._stagnation_counter: int = 0
        self.total_trials: int = 0
        self.total_unique_defects: int = 0
        self._cluster_map: Dict[int, ClusterState] = {cs.cluster_id: cs for cs in clusters}
        self._u1_requeue_decay: float = 0.95
        self._warmup_pool: List[Any] = []
        self._warmup_ptr: int = 0
        self._warmup_total: int = 0
        self._seed_bank: Dict[int, Any] = {}
        self._seed_priority: Dict[int, float] = {}
        self._fallback_pool: List[Any] = []

    def _register_seed(self, seed: Any, priority: float) -> None:
        sid = id(seed)
        self._seed_bank[sid] = seed
        self._seed_priority[sid] = float(priority)

    def initialize_seeds(self, seeds: List[Any], feature_extractor, logits_arr: np.ndarray, true_labels: List[int], keep_count: Optional[int]=None, features_arr: Optional[np.ndarray]=None, feature_batch_size: int=32) -> int:
        initializer = SeedInitializer(clusters=self.clusters, bandwidth=self.bandwidth, tau_r=self.tau_r, top_b_ratio=self.top_b_ratio, enable_vrm=self.enable_vrm)
        primary, fallback = initializer.run(seeds, feature_extractor, logits_arr, true_labels, keep_count=keep_count, features_arr=features_arr, feature_batch_size=feature_batch_size)
        for seed, cluster_id, r_score in primary:
            cs = self._cluster_map.get(cluster_id)
            p_k = cs.priority if cs else 0.0
            init_priority = self.psi1 + p_k * r_score
            self._register_seed(seed, init_priority)
            self._warmup_pool.append(seed)
        for seed, _, _ in fallback:
            self._fallback_pool.append(seed)
        self._warmup_total = len(self._warmup_pool) * 2
        logger.info('[DGFuzzRuntime] Seed init done: Q=%d, Q_V=%d', len(primary), len(fallback))
        return len(primary) + len(fallback)

    def append_direct_seeds(self, seeds: List[Any], features_arr: np.ndarray) -> int:
        features_arr = np.asarray(features_arr, dtype=np.float32)
        if features_arr.ndim != 2:
            raise ValueError(f'features_arr must be 2D, got shape={features_arr.shape}')
        if features_arr.shape[0] != len(seeds):
            raise ValueError(f'features_arr rows ({features_arr.shape[0]}) != seeds ({len(seeds)})')
        added = 0
        for idx, seed in enumerate(seeds):
            fv = features_arr[idx].astype(np.float32).ravel()
            best_id, best_r = self._region_assign(fv)
            if best_r >= self.tau_r:
                cs = self._cluster_map.get(best_id)
                p_k = cs.priority if cs else 0.0
                init_priority = self.psi1 + p_k * max(best_r, _EPS)
                self._register_seed(seed, init_priority)
                self._warmup_pool.append(seed)
            else:
                self._fallback_pool.append(seed)
            added += 1
        self._warmup_total = len(self._warmup_pool) * 2
        logger.info('[DGFuzzRuntime] Direct seed append done: added %d seeds.', added)
        return added

    def select_seed(self, _seed_pool=None) -> Optional[Any]:
        if not self.enable_adaptive_schedule:
            if self._seed_bank:
                seeds = list(self._seed_bank.values())
                idx = int(np.random.randint(0, len(seeds)))
                return seeds[idx]
            if self._warmup_pool:
                idx = int(np.random.randint(0, len(self._warmup_pool)))
                return self._warmup_pool[idx]
            if _seed_pool is not None:
                return _seed_pool.get_next_seed()
            return None
        if self._warmup_ptr < self._warmup_total and self._warmup_pool:
            seed = self._warmup_pool[self._warmup_ptr % len(self._warmup_pool)]
            self._warmup_ptr += 1
            return seed
        if self._should_use_fallback() and self._fallback_pool:
            idx = int(np.random.randint(0, len(self._fallback_pool)))
            return self._fallback_pool[idx]
        if self._seed_priority:
            best_sid = max(self._seed_priority, key=self._seed_priority.get)
            best_seed = self._seed_bank.pop(best_sid, None)
            self._seed_priority.pop(best_sid, None)
            return best_seed
        if _seed_pool is not None:
            return _seed_pool.get_next_seed()
        return None

    def is_interesting(self, feature_vec: np.ndarray, y_true: int, y_pred: int) -> bool:
        if y_true == y_pred:
            return False
        return self._defect_checker.is_unique(feature_vec, y_true, y_pred)

    def update_after_trial(self, seed: Any, feature_vec: np.ndarray, y_true: int, y_pred: int, is_defect: bool, is_unique: bool) -> None:
        self.total_trials += 1
        if not self.enable_adaptive_schedule:
            return
        u_val = self._compute_ut(is_defect, is_unique)
        best_id, best_r = self._region_assign(feature_vec.astype(np.float32).ravel())
        cs = self._cluster_map.get(best_id) if best_r >= self.tau_r else None
        priority = self._compute_priority(u_val, cs, best_r)
        if priority > 0:
            if u_val == 1:
                priority *= self._u1_requeue_decay
            self._register_seed(seed, priority)
        else:
            sid = id(seed)
            if sid in self._seed_priority:
                self._seed_priority[sid] = 0.0
        if cs is not None:
            cs.total_trials += 1
            if is_unique:
                cs.unique_defects += 1
                self.total_unique_defects += 1
            self._update_cluster_priority(cs)
        if is_unique:
            self._stagnation_counter = 0
        else:
            self._stagnation_counter += 1

    def _compute_ut(self, is_defect: bool, is_unique: bool) -> int:
        return 2 if is_defect and is_unique else 1 if is_defect else 0

    def _region_assign(self, feature_vec: np.ndarray) -> Tuple[int, float]:
        if self.enable_vrm:
            return compute_region_strength(feature_vec, self.clusters, self.bandwidth)
        default_cluster_id = self.clusters[0].cluster_id if self.clusters else -1
        return (default_cluster_id, 1.0)

    def _should_use_fallback(self) -> bool:
        m = len(self.clusters)
        if m <= 1:
            return m == 0
        weights = np.array([c.priority for c in self.clusters], dtype=np.float64)
        s = weights.sum()
        if s < _EPS:
            return True
        p = weights / s
        entropy = float(-np.sum(p * np.log(p + _EPS)))
        return entropy / np.log(m) >= self.tau_h

    def _compute_priority(self, u_val: int, cs: Optional[ClusterState], r_score: float) -> float:
        if u_val == 0:
            return 0.0
        base = self.psi2 if u_val == 2 else self.psi1
        if cs is not None and r_score >= self.tau_r:
            return base + cs.priority * r_score
        return base

    def _update_cluster_priority(self, cs: ClusterState) -> None:
        cs.priority = cs.defect_rate
        raw_scores = np.array([c.defect_rate for c in self.clusters], dtype=np.float32)
        if raw_scores.sum() < _EPS:
            uniform = 1.0 / max(len(self.clusters), 1)
            for c in self.clusters:
                c.priority = uniform
            return
        raw_scores -= raw_scores.max()
        exp_scores = np.exp(raw_scores)
        softmax_scores = exp_scores / (exp_scores.sum() + _EPS)
        for c, p in zip(self.clusters, softmax_scores):
            c.priority = float(p)

def build_runtime(adv_library_path: str, cluster_labels: np.ndarray, cluster_centers: Optional[np.ndarray]=None, bandwidth: float=BANDWIDTH_BY_DATASET['cifar10'], tau_r: float=TAU_R, psi1: float=PSI1, psi2: float=PSI2, cosine_threshold: float=COSINE_THRESHOLD_BY_DATASET['cifar10'], top_b_ratio: float=TOP_B_RATIO, tau_h: float=TAU_H, enable_vrm: bool=True, enable_adaptive_schedule: bool=True) -> DGFuzzRuntimeCore:
    data = np.load(adv_library_path, allow_pickle=True)
    adv_features: np.ndarray = data['features']
    unique_cluster_ids = sorted(set(cluster_labels.tolist()) - {-1})
    if len(unique_cluster_ids) == 0:
        raise ValueError('All cluster labels are noise (-1); check clustering parameters in cluster.py.')
    clusters: List[ClusterState] = []
    for enum_idx, k in enumerate(unique_cluster_ids):
        mask = cluster_labels == k
        cluster_feats = adv_features[mask]
        size = int(mask.sum())
        if cluster_centers is not None:
            center = cluster_centers[enum_idx].astype(np.float32)
        else:
            center = cluster_feats.mean(axis=0).astype(np.float32)
        clusters.append(ClusterState(cluster_id=k, center=center, size=size))
    logger.info('[DGFuzzRuntime] Loaded %d adversarial samples from %s; found %d vulnerability clusters.', len(adv_features), adv_library_path, len(clusters))
    return DGFuzzRuntimeCore(clusters=clusters, bandwidth=bandwidth, tau_r=tau_r, psi1=psi1, psi2=psi2, cosine_threshold=cosine_threshold, top_b_ratio=top_b_ratio, tau_h=tau_h, enable_vrm=enable_vrm, enable_adaptive_schedule=enable_adaptive_schedule)

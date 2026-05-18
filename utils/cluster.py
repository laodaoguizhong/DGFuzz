import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import umap
import hdbscan
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

class ClusteringAnalyzer:

    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.device = config.device
        self.model.eval()

    def extract_features(self, samples, batch_size=None):
        if batch_size is None:
            batch_size = 512
        features_list = []
        with torch.no_grad():
            for i in range(0, len(samples), batch_size):
                batch = torch.FloatTensor(samples[i:i + batch_size]).to(self.device)
                if hasattr(self.model, 'get_features'):
                    batch_features = self.model.get_features(batch)
                else:
                    batch_features = self._extract_features_with_hook(batch)
                features_list.append(batch_features.cpu().numpy())
                del batch, batch_features
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        features = np.concatenate(features_list, axis=0)
        features_norm = np.linalg.norm(features, axis=1, keepdims=True)
        features = features / (features_norm + 1e-10)
        return features

    def _logits_from_penultimate(self, feats: torch.Tensor) -> torch.Tensor:
        m = self.model
        if hasattr(m, 'resnet') and hasattr(m.resnet, 'fc'):
            return m.resnet.fc(feats)
        if hasattr(m, 'fc') and isinstance(m.fc, torch.nn.Linear):
            return m.fc(feats)
        c = getattr(m, 'classifier', None)
        if isinstance(c, torch.nn.Linear):
            return c(feats)
        if isinstance(c, torch.nn.Sequential):
            if len(c) >= 7:
                return c[6](feats)
            return c(feats)
        if hasattr(m, 'fc3') and isinstance(m.fc3, torch.nn.Linear):
            return m.fc3(feats)
        if hasattr(m, 'fc1') and isinstance(m.fc1, torch.nn.Linear):
            return m.fc1(feats)
        raise RuntimeError('Cannot infer classifier head from penultimate features; ensure the model implements get_features.')

    def _attach_pool_hook(self):
        captured = []

        def hook_fn(module, input, output):
            if isinstance(output, torch.Tensor):
                captured.append(output)
        hook_handle = None
        for name, module in self.model.named_modules():
            if any((k in name for k in ('avgpool', 'adaptive', 'pool', 'flatten'))):
                hook_handle = module.register_forward_hook(hook_fn)
                break
        if hook_handle is None:
            modules = list(self.model.children())
            if len(modules) > 1:
                hook_handle = modules[-2].register_forward_hook(hook_fn)
        return (captured, hook_handle)

    @staticmethod
    def _flatten_hook_tensor(feat: torch.Tensor) -> torch.Tensor:
        if len(feat.shape) == 4:
            feat = torch.nn.functional.adaptive_avg_pool2d(feat, (1, 1))
        return feat.view(feat.size(0), -1)

    def _forward_logits_and_raw_features_hook(self, x: torch.Tensor):
        captured, hook_handle = self._attach_pool_hook()
        logits = self.model(x)
        if hook_handle:
            hook_handle.remove()
        feat = self._flatten_hook_tensor(captured[0]) if captured else logits
        return (logits, feat)

    @torch.no_grad()
    def forward_logits_and_features(self, x: torch.Tensor):
        x = x.to(self.device)
        if hasattr(self.model, 'get_features'):
            feats = self.model.get_features(x)
            try:
                logits_t = self._logits_from_penultimate(feats)
            except Exception:
                logits_t, feats = self._forward_logits_and_raw_features_hook(x)
        else:
            logits_t, feats = self._forward_logits_and_raw_features_hook(x)
        logits_np = logits_t.detach().cpu().numpy().astype(np.float32)
        feat_np = feats.detach().cpu().numpy().astype(np.float32)
        norms = np.linalg.norm(feat_np, axis=1, keepdims=True)
        feat_np = feat_np / (norms + 1e-10)
        return (logits_np, feat_np)

    def _extract_features_with_hook(self, x):
        captured, hook_handle = self._attach_pool_hook()
        _ = self.model(x)
        if hook_handle:
            hook_handle.remove()
        if captured:
            return self._flatten_hook_tensor(captured[0])
        return self.model(x)

    def perform_umap_reduction(self, features, n_components=2):
        reducer = umap.UMAP(n_neighbors=self.config.umap_n_neighbors, min_dist=self.config.umap_min_dist, n_components=n_components, random_state=self.config.seed, n_jobs=-1, verbose=False)
        features_2d = reducer.fit_transform(features)
        return features_2d

    def perform_hdbscan_clustering_highdim(self, features, min_cluster_size=None, min_samples=None):
        if min_cluster_size is None:
            min_cluster_size = getattr(self.config, 'hdbscan_min_cluster_size', 50)
        if min_samples is None:
            min_samples = getattr(self.config, 'hdbscan_min_samples', 10)
        clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples, cluster_selection_epsilon=0.0, metric='euclidean')
        labels = clusterer.fit_predict(features)
        mask = labels != -1
        valid_labels = labels[mask]
        valid_features = features[mask]
        n_clusters = len(np.unique(valid_labels))
        n_noise = np.sum(labels == -1)
        if n_clusters > 1 and len(valid_features) > n_clusters:
            silhouette = silhouette_score(valid_features, valid_labels)
            davies_bouldin = davies_bouldin_score(valid_features, valid_labels)
            calinski_harabasz = calinski_harabasz_score(valid_features, valid_labels)
        else:
            silhouette = davies_bouldin = calinski_harabasz = 0.0
        if n_clusters > 0:
            cluster_sizes = np.bincount(valid_labels)
        else:
            cluster_sizes = np.array([])
        results = {'clusterer': clusterer, 'labels': labels, 'valid_labels': valid_labels, 'n_clusters': n_clusters, 'n_noise': n_noise, 'silhouette_score': silhouette, 'davies_bouldin_score': davies_bouldin, 'calinski_harabasz_score': calinski_harabasz, 'cluster_sizes': cluster_sizes}
        return results

    def visualize_clustering(self, features_2d, labels, attack_methods, save_path):
        valid_mask = labels != -1
        features_2d_valid = features_2d[valid_mask]
        labels_valid = labels[valid_mask]
        attack_methods_valid = attack_methods[valid_mask]
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4.5))
        all_attacks = np.sort(np.unique(attack_methods))
        if len(all_attacks) <= 10:
            attack_colors = plt.cm.tab10(np.linspace(0, 1, len(all_attacks)))
        else:
            attack_colors = plt.cm.tab20(np.linspace(0, 1, len(all_attacks)))
        attack_color_map = {atk: attack_colors[i] for i, atk in enumerate(all_attacks)}
        for attack in all_attacks:
            mask = attack_methods_valid == attack
            color = attack_color_map[attack]
            if np.any(mask):
                ax1.scatter(features_2d_valid[mask, 0], features_2d_valid[mask, 1], c=[color], label=attack, alpha=0.6, s=10)
            else:
                ax1.scatter([], [], c=[color], label=attack, alpha=0.6, s=10)
        ax1.set_ylabel('Colored by Attack Method', fontsize=14, fontweight='bold')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=12, prop={'weight': 'bold'})
        ax1.grid(True, alpha=0.3)
        ax1.set_xticks([])
        ax1.set_yticks([])
        unique_labels = np.unique(labels_valid)
        n_clusters = len(unique_labels)
        colors2 = plt.cm.Set3(np.linspace(0, 1, n_clusters))
        color_map = {}
        for i, label in enumerate(unique_labels):
            color_map[label] = colors2[i]
        for label in unique_labels:
            mask = labels_valid == label
            label_name = f'Cluster {label}'
            ax2.scatter(features_2d_valid[mask, 0], features_2d_valid[mask, 1], c=[color_map[label]], label=label_name, alpha=0.6, s=10)
        ax2.set_ylabel('Colored by Cluster labels', fontsize=14, fontweight='bold')
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=12, prop={'weight': 'bold'})
        ax2.grid(True, alpha=0.3)
        ax2.set_xticks([])
        ax2.set_yticks([])
        plt.tight_layout()
        plt.savefig(save_path, dpi=900, bbox_inches='tight')
        plt.close()

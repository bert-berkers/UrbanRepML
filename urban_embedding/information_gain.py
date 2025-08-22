#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Information Theory Metrics for Spatial Representation Learning

Implements entropy, mutual information, and information gain calculations
for AlphaEarth embeddings and active inference analysis.
"""

import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass
import logging
from scipy.stats import entropy as scipy_entropy
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics import mutual_info_score
from sklearn.feature_selection import mutual_info_regression
import h3
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


@dataclass
class InformationMetrics:
    """Container for information theory metrics"""
    entropy: float
    conditional_entropy: float
    mutual_information: float
    information_gain: float
    normalized_mi: float
    joint_entropy: float
    kl_divergence: float
    js_divergence: float


class SpatialInformationCalculator:
    """
    Calculate information-theoretic metrics for spatial data.
    """
    
    def __init__(
        self,
        method: str = 'histogram',
        n_bins: int = 50,
        normalize: bool = True,
        epsilon: float = 1e-10
    ):
        """
        Initialize calculator.
        
        Args:
            method: Method for entropy calculation ('histogram', 'kde', 'knn')
            n_bins: Number of bins for histogram method
            normalize: Whether to normalize metrics to [0, 1]
            epsilon: Small value to avoid log(0)
        """
        self.method = method
        self.n_bins = n_bins
        self.normalize = normalize
        self.epsilon = epsilon
    
    def calculate_entropy(
        self,
        data: np.ndarray,
        base: Optional[float] = None
    ) -> float:
        """
        Calculate Shannon entropy H(X) = -sum(p(x) * log(p(x))).
        
        Args:
            data: Data array [n_samples, n_features]
            base: Logarithm base (None for natural log)
            
        Returns:
            Entropy value
        """
        if len(data.shape) == 1:
            data = data.reshape(-1, 1)
        
        if self.method == 'histogram' or self.method == 'shannon':
            # Discretize continuous data
            entropy_values = []
            for col in range(data.shape[1]):
                hist, _ = np.histogram(data[:, col], bins=self.n_bins)
                hist = hist + self.epsilon  # Avoid log(0)
                prob = hist / hist.sum()
                entropy_values.append(scipy_entropy(prob, base=base))
            
            return np.mean(entropy_values)
        
        elif self.method == 'kde':
            # Kernel density estimation (more accurate for continuous)
            from scipy.stats import gaussian_kde
            entropy_values = []
            
            for col in range(data.shape[1]):
                kde = gaussian_kde(data[:, col])
                samples = kde.resample(1000).flatten()
                hist, _ = np.histogram(samples, bins=self.n_bins)
                prob = hist / hist.sum()
                entropy_values.append(scipy_entropy(prob, base=base))
            
            return np.mean(entropy_values)
        
        elif self.method == 'knn':
            # k-nearest neighbor entropy estimation (Kraskov et al.)
            return self._knn_entropy(data)
        
        else:
            raise ValueError(f"Unknown method: {self.method}")
    
    def _knn_entropy(self, data: np.ndarray, k: int = 3) -> float:
        """
        k-nearest neighbor entropy estimator.
        
        Args:
            data: Data array [n_samples, n_features]
            k: Number of nearest neighbors
            
        Returns:
            Entropy estimate
        """
        n, d = data.shape
        
        # Calculate distances
        distances = squareform(pdist(data))
        np.fill_diagonal(distances, np.inf)
        
        # Get k-th nearest neighbor distance for each point
        knn_distances = np.sort(distances, axis=1)[:, k-1]
        
        # Kraskov estimator
        entropy = -np.mean(np.log(knn_distances + self.epsilon)) * d
        entropy += np.log(n - 1) + np.log(np.pi**(d/2) / np.math.gamma(d/2 + 1))
        
        return float(entropy)
    
    def calculate_mutual_information(
        self,
        x: np.ndarray,
        y: np.ndarray,
        method: str = 'kraskov'
    ) -> float:
        """
        Calculate mutual information I(X;Y) = H(X) + H(Y) - H(X,Y).
        
        Args:
            x: First variable [n_samples, n_features_x]
            y: Second variable [n_samples, n_features_y]
            method: Method ('kraskov', 'histogram', 'sklearn')
            
        Returns:
            Mutual information value
        """
        if x.shape[0] != y.shape[0]:
            raise ValueError("x and y must have same number of samples")
        
        if method == 'histogram':
            # Discretize and use histogram-based MI
            x_discrete = np.digitize(x.flatten(), bins=np.linspace(x.min(), x.max(), self.n_bins))
            y_discrete = np.digitize(y.flatten(), bins=np.linspace(y.min(), y.max(), self.n_bins))
            mi = mutual_info_score(x_discrete, y_discrete)
            
        elif method == 'sklearn':
            # Use sklearn's MI regression (for continuous variables)
            if len(x.shape) == 1:
                x = x.reshape(-1, 1)
            if len(y.shape) > 1:
                y = y.mean(axis=1)  # Reduce to 1D for sklearn
            mi = mutual_info_regression(x, y, random_state=42)[0]
            
        elif method == 'kraskov':
            # Kraskov-Stögbauer-Grassberger estimator
            mi = self._kraskov_mi(x, y)
            
        else:
            # Formula: I(X;Y) = H(X) + H(Y) - H(X,Y)
            h_x = self.calculate_entropy(x)
            h_y = self.calculate_entropy(y)
            h_xy = self.calculate_entropy(np.concatenate([x, y], axis=1))
            mi = h_x + h_y - h_xy
        
        if self.normalize:
            # Normalized MI: I(X;Y) / sqrt(H(X) * H(Y))
            h_x = self.calculate_entropy(x)
            h_y = self.calculate_entropy(y)
            if h_x > 0 and h_y > 0:
                mi = mi / np.sqrt(h_x * h_y)
        
        return float(mi)
    
    def _kraskov_mi(self, x: np.ndarray, y: np.ndarray, k: int = 3) -> float:
        """
        Kraskov-Stögbauer-Grassberger mutual information estimator.
        
        Args:
            x: First variable [n_samples, n_features_x]
            y: Second variable [n_samples, n_features_y]
            k: Number of nearest neighbors
            
        Returns:
            Mutual information estimate
        """
        n = x.shape[0]
        
        if len(x.shape) == 1:
            x = x.reshape(-1, 1)
        if len(y.shape) == 1:
            y = y.reshape(-1, 1)
        
        # Combine for joint space
        xy = np.concatenate([x, y], axis=1)
        
        # Calculate distances in joint space
        distances_joint = squareform(pdist(xy))
        np.fill_diagonal(distances_joint, np.inf)
        
        # Get k-th nearest neighbor distance
        knn_distances = np.sort(distances_joint, axis=1)[:, k-1]
        
        # Count neighbors in marginal spaces
        distances_x = squareform(pdist(x))
        distances_y = squareform(pdist(y))
        
        nx = np.sum(distances_x < knn_distances[:, np.newaxis], axis=1)
        ny = np.sum(distances_y < knn_distances[:, np.newaxis], axis=1)
        
        # Kraskov estimator
        psi = lambda x: np.log(x + 1) - 1/(x + 1)  # Digamma approximation
        mi = psi(n) + psi(k) - np.mean(psi(nx + 1) + psi(ny + 1))
        
        return float(mi)
    
    def calculate_information_gain(
        self,
        prior: np.ndarray,
        posterior: np.ndarray
    ) -> float:
        """
        Calculate information gain IG = H(prior) - H(posterior).
        
        Args:
            prior: Prior distribution/embeddings
            posterior: Posterior distribution/embeddings
            
        Returns:
            Information gain value
        """
        h_prior = self.calculate_entropy(prior)
        h_posterior = self.calculate_entropy(posterior)
        return float(h_prior - h_posterior)
    
    def calculate_kl_divergence(
        self,
        p: np.ndarray,
        q: np.ndarray
    ) -> float:
        """
        Calculate Kullback-Leibler divergence D_KL(P||Q).
        
        Args:
            p: True distribution
            q: Approximate distribution
            
        Returns:
            KL divergence value
        """
        # Discretize if continuous
        if self.method == 'histogram':
            # Create histograms
            bins = np.linspace(
                min(p.min(), q.min()),
                max(p.max(), q.max()),
                self.n_bins
            )
            
            p_hist, _ = np.histogram(p.flatten(), bins=bins)
            q_hist, _ = np.histogram(q.flatten(), bins=bins)
            
            # Normalize
            p_hist = p_hist + self.epsilon
            q_hist = q_hist + self.epsilon
            p_prob = p_hist / p_hist.sum()
            q_prob = q_hist / q_hist.sum()
            
            # KL divergence
            kl = np.sum(p_prob * np.log(p_prob / q_prob))
        else:
            # For continuous, use Gaussian assumption
            p_mean, p_std = p.mean(), p.std() + self.epsilon
            q_mean, q_std = q.mean(), q.std() + self.epsilon
            
            kl = np.log(q_std / p_std) + (p_std**2 + (p_mean - q_mean)**2) / (2 * q_std**2) - 0.5
        
        return float(kl)
    
    def calculate_js_divergence(
        self,
        p: np.ndarray,
        q: np.ndarray
    ) -> float:
        """
        Calculate Jensen-Shannon divergence (symmetric KL).
        
        Args:
            p: First distribution
            q: Second distribution
            
        Returns:
            JS divergence value
        """
        # Create mixture distribution
        m = (p + q) / 2
        
        # JS = 0.5 * KL(P||M) + 0.5 * KL(Q||M)
        kl_pm = self.calculate_kl_divergence(p, m)
        kl_qm = self.calculate_kl_divergence(q, m)
        
        return float(0.5 * (kl_pm + kl_qm))
    
    def calculate_all_metrics(
        self,
        x: np.ndarray,
        y: Optional[np.ndarray] = None,
        prior: Optional[np.ndarray] = None
    ) -> InformationMetrics:
        """
        Calculate all information metrics.
        
        Args:
            x: Primary data
            y: Secondary data (for joint metrics)
            prior: Prior distribution (for information gain)
            
        Returns:
            InformationMetrics object with all calculations
        """
        # Single variable metrics
        h_x = self.calculate_entropy(x)
        
        # Initialize with defaults
        metrics = InformationMetrics(
            entropy=h_x,
            conditional_entropy=0.0,
            mutual_information=0.0,
            information_gain=0.0,
            normalized_mi=0.0,
            joint_entropy=0.0,
            kl_divergence=0.0,
            js_divergence=0.0
        )
        
        # Joint metrics if y provided
        if y is not None:
            h_y = self.calculate_entropy(y)
            h_xy = self.calculate_entropy(np.concatenate([x, y], axis=1))
            mi = self.calculate_mutual_information(x, y)
            
            metrics.joint_entropy = h_xy
            metrics.mutual_information = mi
            metrics.conditional_entropy = h_xy - h_y  # H(X|Y)
            metrics.normalized_mi = mi / np.sqrt(h_x * h_y) if h_x > 0 and h_y > 0 else 0
            metrics.kl_divergence = self.calculate_kl_divergence(x, y)
            metrics.js_divergence = self.calculate_js_divergence(x, y)
        
        # Information gain if prior provided
        if prior is not None:
            metrics.information_gain = self.calculate_information_gain(prior, x)
        
        return metrics


class SpatialInformationGain:
    """
    Calculate information gain for spatial H3 hexagons.
    """
    
    def __init__(
        self,
        h3_resolution: int = 8,
        temporal_window: int = 1
    ):
        self.h3_resolution = h3_resolution
        self.temporal_window = temporal_window
        self.calculator = SpatialInformationCalculator()
    
    def calculate_hexagon_information(
        self,
        embeddings: Dict[str, np.ndarray]
    ) -> Dict[str, float]:
        """
        Calculate information content for each hexagon.
        
        Args:
            embeddings: Dictionary mapping H3 index to embedding vector
            
        Returns:
            Dictionary mapping H3 index to information content
        """
        information = {}
        
        for h3_idx, embedding in embeddings.items():
            # Calculate entropy as measure of information content
            entropy = self.calculator.calculate_entropy(embedding.reshape(1, -1))
            information[h3_idx] = entropy
        
        return information
    
    def calculate_spatial_mutual_information(
        self,
        embeddings: Dict[str, np.ndarray],
        h3_index: str,
        neighbor_ring: int = 1
    ) -> float:
        """
        Calculate mutual information between hexagon and neighbors.
        
        Args:
            embeddings: Dictionary mapping H3 index to embedding
            h3_index: Central hexagon
            neighbor_ring: Ring distance for neighbors
            
        Returns:
            Average mutual information with neighbors
        """
        if h3_index not in embeddings:
            return 0.0
        
        center_embedding = embeddings[h3_index]
        neighbors = set(h3.grid_disk(h3_index, neighbor_ring))
        neighbors.discard(h3_index)
        
        mi_values = []
        for neighbor in neighbors:
            if neighbor in embeddings:
                mi = self.calculator.calculate_mutual_information(
                    center_embedding.reshape(1, -1),
                    embeddings[neighbor].reshape(1, -1)
                )
                mi_values.append(mi)
        
        return np.mean(mi_values) if mi_values else 0.0
    
    def calculate_temporal_information_gain(
        self,
        embeddings_t1: Dict[str, np.ndarray],
        embeddings_t2: Dict[str, np.ndarray]
    ) -> Dict[str, float]:
        """
        Calculate information gain between time steps.
        
        Args:
            embeddings_t1: Embeddings at time t
            embeddings_t2: Embeddings at time t+1
            
        Returns:
            Dictionary mapping H3 index to temporal information gain
        """
        information_gain = {}
        
        common_indices = set(embeddings_t1.keys()) & set(embeddings_t2.keys())
        
        for h3_idx in common_indices:
            ig = self.calculator.calculate_information_gain(
                embeddings_t1[h3_idx].reshape(1, -1),
                embeddings_t2[h3_idx].reshape(1, -1)
            )
            information_gain[h3_idx] = ig
        
        return information_gain
    
    def identify_high_information_gaps(
        self,
        embeddings: Dict[str, np.ndarray],
        coverage_map: Dict[str, bool],
        threshold_percentile: float = 90
    ) -> List[str]:
        """
        Identify gaps with high expected information gain.
        
        Args:
            embeddings: Current embeddings
            coverage_map: Map of covered (True) vs gap (False) hexagons
            threshold_percentile: Percentile for high information
            
        Returns:
            List of H3 indices with high information gaps
        """
        # Calculate information for existing hexagons
        information = self.calculate_hexagon_information(embeddings)
        
        # Find threshold
        values = list(information.values())
        if not values:
            return []
        
        threshold = np.percentile(values, threshold_percentile)
        
        # Identify high-information gaps
        high_info_gaps = []
        
        for h3_idx, is_covered in coverage_map.items():
            if not is_covered:  # It's a gap
                # Check information of neighbors
                neighbors = set(h3.grid_disk(h3_idx, 1))
                neighbor_info = [
                    information.get(n, 0) 
                    for n in neighbors 
                    if n in information
                ]
                
                if neighbor_info:
                    avg_neighbor_info = np.mean(neighbor_info)
                    if avg_neighbor_info >= threshold:
                        high_info_gaps.append(h3_idx)
        
        return high_info_gaps


class InformationBottleneck:
    """
    Information Bottleneck method for dimensionality reduction.
    Compress data while preserving relevant information.
    """
    
    def __init__(
        self,
        compression_dim: int = 16,
        beta: float = 1.0,
        n_iterations: int = 100
    ):
        """
        Initialize Information Bottleneck.
        
        Args:
            compression_dim: Compressed dimension
            beta: Trade-off between compression and preservation
            n_iterations: Number of optimization iterations
        """
        self.compression_dim = compression_dim
        self.beta = beta
        self.n_iterations = n_iterations
        self.encoder = None
        self.decoder = None
    
    def fit(self, x: np.ndarray, y: Optional[np.ndarray] = None):
        """
        Fit the information bottleneck.
        
        Args:
            x: Input data [n_samples, n_features]
            y: Target variable (optional)
        """
        from sklearn.decomposition import PCA
        
        # Simple implementation using PCA as proxy
        # In practice, would use iterative optimization
        self.encoder = PCA(n_components=self.compression_dim)
        self.encoder.fit(x)
        
        # Store decoder as pseudo-inverse
        self.decoder = self.encoder.components_.T
    
    def compress(self, x: np.ndarray) -> np.ndarray:
        """
        Compress data through bottleneck.
        
        Args:
            x: Input data
            
        Returns:
            Compressed representation
        """
        if self.encoder is None:
            raise ValueError("Must fit before compressing")
        
        return self.encoder.transform(x)
    
    def reconstruct(self, z: np.ndarray) -> np.ndarray:
        """
        Reconstruct from compressed representation.
        
        Args:
            z: Compressed data
            
        Returns:
            Reconstructed data
        """
        if self.decoder is None:
            raise ValueError("Must fit before reconstructing")
        
        return z @ self.decoder.T
    
    def calculate_information_plane(
        self,
        x: np.ndarray,
        y: np.ndarray
    ) -> Tuple[float, float]:
        """
        Calculate position in information plane.
        
        Args:
            x: Input data
            y: Target data
            
        Returns:
            (I(X;T), I(T;Y)) where T is compressed representation
        """
        calculator = SpatialInformationCalculator()
        
        # Compress
        t = self.compress(x)
        
        # Calculate mutual information
        i_xt = calculator.calculate_mutual_information(x, t)
        i_ty = calculator.calculate_mutual_information(t, y) if y is not None else 0
        
        return i_xt, i_ty
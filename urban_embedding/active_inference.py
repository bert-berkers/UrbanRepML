#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Active Inference Module for Spatial Representation Learning

Implements free energy principle and active inference for geospatial analysis,
following Friston's framework with adaptations for H3 hexagonal grids.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
import logging
from scipy.special import digamma, gammaln
from scipy.stats import entropy
import h3

logger = logging.getLogger(__name__)


@dataclass
class FreeEnergyComponents:
    """Components of variational free energy calculation"""
    total: float
    complexity: float  # KL divergence between posterior and prior
    accuracy: float    # Negative log-likelihood of observations
    entropy: float     # Entropy of the posterior
    expected: float    # Expected free energy for planning


class ActiveInferenceModule(nn.Module):
    """
    Active Inference implementation for spatial data.
    
    Implements variational free energy minimization for belief updating
    and expected free energy for action selection (gap detection).
    """
    
    def __init__(
        self,
        input_dim: int = 64,
        hidden_dims: List[int] = [128, 64, 32],
        latent_dim: int = 16,
        precision_init: float = 1.0,
        complexity_weight: float = 0.5,
        accuracy_weight: float = 0.5,
        device: str = 'cuda'  # RTX 3090 POWER!
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.precision = nn.Parameter(torch.tensor(precision_init))
        self.complexity_weight = complexity_weight
        self.accuracy_weight = accuracy_weight
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # Build encoder (recognition model q(s|o))
        encoder_layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            prev_dim = hidden_dim
        
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Latent space parameters
        self.mu_encoder = nn.Linear(prev_dim, latent_dim)
        self.logvar_encoder = nn.Linear(prev_dim, latent_dim)
        
        # Build decoder (generative model p(o|s))
        decoder_layers = []
        prev_dim = latent_dim
        for hidden_dim in reversed(hidden_dims):
            decoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            prev_dim = hidden_dim
        
        decoder_layers.append(nn.Linear(prev_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)
        
        # Prior parameters (p(s))
        self.register_buffer('prior_mu', torch.zeros(latent_dim))
        self.register_buffer('prior_logvar', torch.zeros(latent_dim))
        
        self.to(self.device)
    
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode observations to posterior parameters q(s|o).
        
        Args:
            x: Observations [batch_size, input_dim]
            
        Returns:
            mu: Mean of posterior [batch_size, latent_dim]
            logvar: Log variance of posterior [batch_size, latent_dim]
        """
        h = self.encoder(x)
        mu = self.mu_encoder(h)
        logvar = self.logvar_encoder(h)
        return mu, logvar
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization trick for differentiable sampling.
        
        Args:
            mu: Mean [batch_size, latent_dim]
            logvar: Log variance [batch_size, latent_dim]
            
        Returns:
            z: Sampled latent state [batch_size, latent_dim]
        """
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            return mu
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode latent states to observations p(o|s).
        
        Args:
            z: Latent states [batch_size, latent_dim]
            
        Returns:
            x_recon: Reconstructed observations [batch_size, input_dim]
        """
        return self.decoder(z)
    
    def calculate_free_energy(
        self,
        x: torch.Tensor,
        x_recon: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor
    ) -> FreeEnergyComponents:
        """
        Calculate variational free energy F = D_KL[q(s)||p(s)] - E_q[log p(o|s)].
        
        Args:
            x: Original observations [batch_size, input_dim]
            x_recon: Reconstructed observations [batch_size, input_dim]
            mu: Posterior mean [batch_size, latent_dim]
            logvar: Posterior log variance [batch_size, latent_dim]
            
        Returns:
            FreeEnergyComponents with all terms
        """
        batch_size = x.shape[0]
        
        # Complexity: KL divergence between posterior and prior
        # KL[q(s|o)||p(s)] = 0.5 * sum(1 + logvar - mu^2 - exp(logvar))
        kl_div = -0.5 * torch.sum(
            1 + logvar - self.prior_logvar - mu.pow(2) - logvar.exp()
        ) / batch_size
        
        # Accuracy: Negative log-likelihood (reconstruction error)
        # Using Gaussian likelihood with learned precision
        recon_loss = F.mse_loss(x_recon, x, reduction='sum') / batch_size
        log_likelihood = -0.5 * self.precision * recon_loss
        
        # Posterior entropy for monitoring
        posterior_entropy = 0.5 * torch.sum(logvar + np.log(2 * np.pi * np.e)) / batch_size
        
        # Total free energy
        free_energy = self.complexity_weight * kl_div - self.accuracy_weight * log_likelihood
        
        # Expected free energy (for active inference planning)
        # G = epistemic value + pragmatic value
        # Simplified: use uncertainty (entropy) as epistemic value
        expected_free_energy = posterior_entropy.item() - log_likelihood.item()
        
        return FreeEnergyComponents(
            total=free_energy.item(),
            complexity=kl_div.item(),
            accuracy=-log_likelihood.item(),
            entropy=posterior_entropy.item(),
            expected=expected_free_energy
        )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, FreeEnergyComponents]:
        """
        Forward pass through active inference model.
        
        Args:
            x: Observations [batch_size, input_dim]
            
        Returns:
            x_recon: Reconstructed observations
            mu: Posterior mean
            logvar: Posterior log variance
            free_energy: Free energy components
        """
        # Encode to posterior
        mu, logvar = self.encode(x)
        
        # Sample latent state
        z = self.reparameterize(mu, logvar)
        
        # Decode to reconstruction
        x_recon = self.decode(z)
        
        # Calculate free energy
        free_energy = self.calculate_free_energy(x, x_recon, mu, logvar)
        
        return x_recon, mu, logvar, free_energy
    
    def update_beliefs(
        self,
        observations: torch.Tensor,
        prior_mu: Optional[torch.Tensor] = None,
        prior_logvar: Optional[torch.Tensor] = None,
        learning_rate: float = 0.01
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Update beliefs using gradient descent on free energy.
        
        Args:
            observations: New observations [batch_size, input_dim]
            prior_mu: Prior mean (optional)
            prior_logvar: Prior log variance (optional)
            learning_rate: Step size for belief updating
            
        Returns:
            Updated posterior mean and log variance
        """
        # Set prior if provided
        if prior_mu is not None:
            self.prior_mu = prior_mu
        if prior_logvar is not None:
            self.prior_logvar = prior_logvar
        
        # Initialize posterior with prior
        mu = self.prior_mu.clone().requires_grad_(True)
        logvar = self.prior_logvar.clone().requires_grad_(True)
        
        # Gradient descent on free energy
        optimizer = torch.optim.Adam([mu, logvar], lr=learning_rate)
        
        for _ in range(10):  # Fixed number of iterations
            optimizer.zero_grad()
            
            # Sample and decode
            z = self.reparameterize(mu, logvar)
            x_recon = self.decode(z)
            
            # Calculate free energy
            free_energy_components = self.calculate_free_energy(
                observations, x_recon, mu, logvar
            )
            
            # Backpropagate
            loss = torch.tensor(free_energy_components.total, requires_grad=True)
            loss.backward()
            optimizer.step()
        
        return mu.detach(), logvar.detach()
    
    def calculate_expected_free_energy(
        self,
        observations: torch.Tensor,
        policies: Optional[torch.Tensor] = None,
        preferences: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Calculate expected free energy for policy selection.
        G = epistemic value + pragmatic value
        
        Args:
            observations: Current observations [batch_size, input_dim]
            policies: Possible policies/actions (optional)
            preferences: Preferred observations (optional)
            
        Returns:
            Expected free energy for each policy
        """
        mu, logvar = self.encode(observations)
        
        # Epistemic value: information gain (negative entropy)
        epistemic_value = -0.5 * torch.sum(logvar + np.log(2 * np.pi * np.e), dim=1)
        
        # Pragmatic value: distance to preferences
        if preferences is not None:
            z = self.reparameterize(mu, logvar)
            predicted = self.decode(z)
            pragmatic_value = -F.mse_loss(predicted, preferences, reduction='none').sum(dim=1)
        else:
            pragmatic_value = torch.zeros_like(epistemic_value)
        
        # Total expected free energy
        expected_free_energy = epistemic_value + pragmatic_value
        
        return expected_free_energy


class SpatialMarkovBlanket:
    """
    Implements Markov blankets for spatial active inference on H3 grid.
    """
    
    def __init__(
        self,
        h3_resolution: int = 8,
        spatial_radius: int = 2,
        include_diagonal: bool = True
    ):
        self.h3_resolution = h3_resolution
        self.spatial_radius = spatial_radius
        self.include_diagonal = include_diagonal
    
    def get_markov_blanket(self, h3_index: str) -> Dict[str, List[str]]:
        """
        Get Markov blanket for an H3 cell.
        
        Args:
            h3_index: H3 index of the cell
            
        Returns:
            Dictionary with 'parents', 'children', 'blanket' keys
        """
        # Get neighbors at different distances
        blanket = set()
        
        for ring in range(1, self.spatial_radius + 1):
            neighbors = set(h3.grid_disk(h3_index, ring))
            blanket.update(neighbors)
        
        # Remove the cell itself
        blanket.discard(h3_index)
        
        # Hierarchical relationships
        parent = h3.cell_to_parent(h3_index, self.h3_resolution - 1)
        children = h3.cell_to_children(h3_index, self.h3_resolution + 1)
        
        return {
            'parents': [parent],
            'children': list(children),
            'blanket': list(blanket)
        }
    
    def calculate_conditional_independence(
        self,
        embeddings: Dict[str, np.ndarray],
        h3_index: str
    ) -> float:
        """
        Calculate conditional independence given Markov blanket.
        
        Args:
            embeddings: Dictionary mapping H3 indices to embedding vectors
            h3_index: H3 index to calculate for
            
        Returns:
            Conditional independence score
        """
        if h3_index not in embeddings:
            return 0.0
        
        blanket_info = self.get_markov_blanket(h3_index)
        blanket_indices = blanket_info['blanket']
        
        # Get embeddings for cell and blanket
        cell_embedding = embeddings[h3_index]
        blanket_embeddings = np.array([
            embeddings[idx] for idx in blanket_indices 
            if idx in embeddings
        ])
        
        if len(blanket_embeddings) == 0:
            return 1.0  # Fully independent if no blanket
        
        # Calculate conditional entropy H(cell|blanket)
        # Simplified: use correlation as proxy
        correlations = np.corrcoef(
            cell_embedding.reshape(1, -1),
            blanket_embeddings
        )[0, 1:]
        
        # Independence score: 1 - mean absolute correlation
        independence = 1.0 - np.mean(np.abs(correlations))
        
        return float(independence)


class HierarchicalActiveInference:
    """
    Hierarchical active inference across multiple H3 resolutions.
    """
    
    def __init__(
        self,
        resolutions: List[int] = [5, 6, 7, 8, 9, 10],
        primary_resolution: int = 8,
        coupling_strength: float = 0.3
    ):
        self.resolutions = sorted(resolutions)
        self.primary_resolution = primary_resolution
        self.coupling_strength = coupling_strength
        
        # Create active inference module for each resolution
        self.modules = {}
        for res in resolutions:
            self.modules[res] = ActiveInferenceModule(
                input_dim=64,  # AlphaEarth dimensions
                latent_dim=16 if res == primary_resolution else 8
            )
    
    def hierarchical_free_energy(
        self,
        observations: Dict[int, torch.Tensor]
    ) -> Dict[int, FreeEnergyComponents]:
        """
        Calculate free energy at each hierarchical level.
        
        Args:
            observations: Dictionary mapping resolution to observations
            
        Returns:
            Dictionary mapping resolution to free energy components
        """
        free_energies = {}
        
        for res in self.resolutions:
            if res not in observations:
                continue
            
            module = self.modules[res]
            _, _, _, fe = module(observations[res])
            free_energies[res] = fe
        
        return free_energies
    
    def cross_resolution_consistency(
        self,
        embeddings: Dict[int, torch.Tensor]
    ) -> float:
        """
        Calculate consistency across resolutions.
        
        Args:
            embeddings: Dictionary mapping resolution to embeddings
            
        Returns:
            Consistency score
        """
        if len(embeddings) < 2:
            return 1.0
        
        consistencies = []
        resolutions = sorted(embeddings.keys())
        
        for i in range(len(resolutions) - 1):
            res1, res2 = resolutions[i], resolutions[i + 1]
            
            # Project to common space and calculate similarity
            emb1 = F.normalize(embeddings[res1], dim=-1)
            emb2 = F.normalize(embeddings[res2], dim=-1)
            
            # Resize if needed (simplified)
            if emb1.shape != emb2.shape:
                min_size = min(emb1.shape[0], emb2.shape[0])
                emb1 = emb1[:min_size]
                emb2 = emb2[:min_size]
            
            similarity = F.cosine_similarity(emb1, emb2, dim=-1).mean()
            consistencies.append(similarity.item())
        
        return np.mean(consistencies) if consistencies else 1.0
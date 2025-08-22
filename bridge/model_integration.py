"""
Model integration utilities for combining UrbanRepML and GEO-INFER capabilities.
"""

import torch
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from pathlib import Path


def combine_embeddings(
    urbanreml_embeddings: np.ndarray,
    geoinfer_features: np.ndarray,
    combination_method: str = "concatenate"
) -> np.ndarray:
    """
    Combine embeddings from UrbanRepML with GEO-INFER features.
    
    Args:
        urbanreml_embeddings: Embeddings from UrbanRepML models
        geoinfer_features: Features from GEO-INFER modules
        combination_method: Method to combine ('concatenate', 'average', 'weighted')
    
    Returns:
        Combined feature array
    """
    if combination_method == "concatenate":
        return np.concatenate([urbanreml_embeddings, geoinfer_features], axis=1)
    elif combination_method == "average":
        # Ensure compatible dimensions
        min_dim = min(urbanreml_embeddings.shape[1], geoinfer_features.shape[1])
        return (urbanreml_embeddings[:, :min_dim] + geoinfer_features[:, :min_dim]) / 2
    elif combination_method == "weighted":
        # Default weights, can be parameterized
        w1, w2 = 0.6, 0.4
        min_dim = min(urbanreml_embeddings.shape[1], geoinfer_features.shape[1])
        return w1 * urbanreml_embeddings[:, :min_dim] + w2 * geoinfer_features[:, :min_dim]
    else:
        raise ValueError(f"Unknown combination method: {combination_method}")


def active_inference_wrapper(
    urbanreml_model: torch.nn.Module,
    observation: torch.Tensor,
    prior_belief: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Wrap UrbanRepML models with Active Inference principles from GEO-INFER.
    
    Args:
        urbanreml_model: Trained UrbanRepML model
        observation: Current observation tensor
        prior_belief: Prior belief state (optional)
    
    Returns:
        Tuple of (posterior_belief, prediction_error)
    """
    # Generate prediction from model
    with torch.no_grad():
        prediction = urbanreml_model(observation)
    
    # Calculate prediction error (surprise)
    if prior_belief is not None:
        prediction_error = torch.norm(prediction - prior_belief, dim=-1)
    else:
        prediction_error = torch.zeros(prediction.shape[0])
    
    # Update belief (simplified Active Inference update)
    learning_rate = 0.1
    if prior_belief is not None:
        posterior_belief = prior_belief + learning_rate * (prediction - prior_belief)
    else:
        posterior_belief = prediction
    
    return posterior_belief, prediction_error


def multi_resolution_adapter(
    data: Dict[int, pd.DataFrame],
    target_resolution: int = 8
) -> pd.DataFrame:
    """
    Adapt multi-resolution data from UrbanRepML for GEO-INFER processing.
    
    Args:
        data: Dictionary mapping H3 resolutions to DataFrames
        target_resolution: Target H3 resolution for GEO-INFER
    
    Returns:
        Adapted DataFrame at target resolution
    """
    if target_resolution in data:
        base_df = data[target_resolution].copy()
    else:
        # Find closest resolution and adapt
        available_res = list(data.keys())
        closest_res = min(available_res, key=lambda x: abs(x - target_resolution))
        base_df = data[closest_res].copy()
        
        # Add resolution conversion metadata
        base_df['original_resolution'] = closest_res
        base_df['adapted_to'] = target_resolution
    
    # Aggregate features from other resolutions if available
    for res, df in data.items():
        if res != target_resolution:
            # Add aggregated features with resolution suffix
            feature_cols = df.select_dtypes(include='number').columns
            for col in feature_cols:
                base_df[f"{col}_res{res}"] = df[col].reindex(base_df.index, fill_value=0)
    
    return base_df
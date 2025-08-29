"""
AlphaEarth-Conditioned Semantic Segmentation Network.

This implements an attentional U-Net that uses AlphaEarth embeddings
to condition the segmentation of DINOv3 visual features.

Key innovation: Hierarchical conditioning where global AlphaEarth context
improves local DINOv3 segmentation through cross-attention mechanisms.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import numpy as np
from dataclasses import dataclass

from .segmentation_classes import SegmentationClasses, NetherlandsLandCover


@dataclass
class ConditioningConfig:
    """Configuration for AlphaEarth conditioning."""
    alphaearth_dim: int = 64
    dinov3_dim: int = 768
    conditioning_dim: int = 256
    num_conditioning_layers: int = 3
    use_cross_attention: bool = True
    attention_heads: int = 8


class CrossAttentionBlock(nn.Module):
    """
    Cross-attention between DINOv3 features (queries) and AlphaEarth embeddings (keys/values).
    
    This implements the conditioning mechanism where global satellite context
    (AlphaEarth) guides local visual understanding (DINOv3).
    """
    
    def __init__(self, 
                 feature_dim: int,
                 conditioning_dim: int,
                 num_heads: int = 8,
                 dropout: float = 0.1):
        super().__init__()
        
        self.feature_dim = feature_dim
        self.conditioning_dim = conditioning_dim
        self.num_heads = num_heads
        
        # Linear projections
        self.q_proj = nn.Linear(feature_dim, feature_dim)
        self.k_proj = nn.Linear(conditioning_dim, feature_dim)
        self.v_proj = nn.Linear(conditioning_dim, feature_dim)
        self.out_proj = nn.Linear(feature_dim, feature_dim)
        
        # Layer norm and dropout
        self.layer_norm = nn.LayerNorm(feature_dim)
        self.dropout = nn.Dropout(dropout)
        
        # Scale for attention
        self.scale = (feature_dim // num_heads) ** -0.5
    
    def forward(self, features: torch.Tensor, conditioning: torch.Tensor) -> torch.Tensor:
        """
        Apply cross-attention conditioning.
        
        Args:
            features: DINOv3 features (B, N_patches, feature_dim)
            conditioning: AlphaEarth embeddings (B, N_regions, conditioning_dim)
            
        Returns:
            Conditioned features (B, N_patches, feature_dim)
        """
        B, N_patches, _ = features.shape
        B_cond, N_regions, _ = conditioning.shape
        
        assert B == B_cond, "Batch sizes must match"
        
        # Project to query, key, value
        queries = self.q_proj(features)  # (B, N_patches, feature_dim)
        keys = self.k_proj(conditioning)  # (B, N_regions, feature_dim)  
        values = self.v_proj(conditioning)  # (B, N_regions, feature_dim)
        
        # Reshape for multi-head attention
        queries = queries.view(B, N_patches, self.num_heads, -1).transpose(1, 2)
        keys = keys.view(B, N_regions, self.num_heads, -1).transpose(1, 2)
        values = values.view(B, N_regions, self.num_heads, -1).transpose(1, 2)
        
        # Compute attention scores
        attention_scores = torch.matmul(queries, keys.transpose(-2, -1)) * self.scale
        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        attended = torch.matmul(attention_weights, values)
        
        # Reshape back
        attended = attended.transpose(1, 2).contiguous().view(B, N_patches, -1)
        
        # Output projection and residual connection
        output = self.out_proj(attended)
        output = self.layer_norm(features + output)
        
        return output


class FeatureFusionBlock(nn.Module):
    """
    Fuse AlphaEarth and DINOv3 features at multiple scales.
    
    Implements the hierarchical fusion that combines global satellite context
    with local visual details through learnable attention weights.
    """
    
    def __init__(self,
                 alphaearth_dim: int,
                 dinov3_dim: int,
                 fusion_dim: int,
                 num_heads: int = 8):
        super().__init__()
        
        # Project both modalities to common dimension
        self.alphaearth_proj = nn.Sequential(
            nn.Linear(alphaearth_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        self.dinov3_proj = nn.Sequential(
            nn.Linear(dinov3_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Cross-attention for conditioning
        self.cross_attention = CrossAttentionBlock(
            feature_dim=fusion_dim,
            conditioning_dim=fusion_dim,
            num_heads=num_heads
        )
        
        # Fusion layers
        self.fusion_layers = nn.Sequential(
            nn.Linear(fusion_dim * 2, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(fusion_dim, fusion_dim)
        )
    
    def forward(self,
                alphaearth_features: torch.Tensor,
                dinov3_features: torch.Tensor) -> torch.Tensor:
        """
        Fuse AlphaEarth and DINOv3 features.
        
        Args:
            alphaearth_features: (B, N_regions, alphaearth_dim)
            dinov3_features: (B, N_patches, dinov3_dim)
            
        Returns:
            Fused features: (B, N_patches, fusion_dim)
        """
        # Project to common dimension
        alpha_proj = self.alphaearth_proj(alphaearth_features)
        dino_proj = self.dinov3_proj(dinov3_features)
        
        # Apply cross-attention conditioning
        conditioned_dino = self.cross_attention(dino_proj, alpha_proj)
        
        # Interpolate AlphaEarth features to match DINOv3 spatial resolution
        if alpha_proj.shape[1] != conditioned_dino.shape[1]:
            alpha_proj = F.interpolate(
                alpha_proj.transpose(1, 2),
                size=conditioned_dino.shape[1],
                mode='linear'
            ).transpose(1, 2)
        
        # Concatenate and fuse
        combined = torch.cat([conditioned_dino, alpha_proj], dim=-1)
        fused = self.fusion_layers(combined)
        
        return fused


class AlphaEarthConditionedUNet(nn.Module):
    """
    U-Net architecture conditioned on AlphaEarth embeddings.
    
    This implements the attentional U-Net that you described, where:
    1. DINOv3 provides local visual features
    2. AlphaEarth provides global satellite context
    3. Cross-attention conditions segmentation at multiple scales
    4. Hierarchical processing preserves multi-scale information
    """
    
    def __init__(self,
                 config: ConditioningConfig,
                 num_classes: int = None,
                 image_size: int = 512):
        super().__init__()
        
        if num_classes is None:
            num_classes = SegmentationClasses.get_num_classes()
        
        self.config = config
        self.num_classes = num_classes
        self.image_size = image_size
        
        # Feature fusion block
        self.feature_fusion = FeatureFusionBlock(
            alphaearth_dim=config.alphaearth_dim,
            dinov3_dim=config.dinov3_dim,
            fusion_dim=config.conditioning_dim,
            num_heads=config.attention_heads
        )
        
        # U-Net encoder (downsampling path)
        self.encoder_blocks = nn.ModuleList([
            self._make_encoder_block(config.conditioning_dim, 128),
            self._make_encoder_block(128, 256),
            self._make_encoder_block(256, 512),
            self._make_encoder_block(512, 1024)
        ])
        
        # U-Net decoder (upsampling path)
        self.decoder_blocks = nn.ModuleList([
            self._make_decoder_block(1024, 512),
            self._make_decoder_block(512 + 512, 256),
            self._make_decoder_block(256 + 256, 128),
            self._make_decoder_block(128 + 128, 64)
        ])
        
        # Final segmentation head
        self.segmentation_head = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, num_classes, kernel_size=1)
        )
        
        # Conditioning attention at multiple scales
        self.conditioning_layers = nn.ModuleList([
            CrossAttentionBlock(
                feature_dim=dim,
                conditioning_dim=config.conditioning_dim,
                num_heads=config.attention_heads // 2
            ) for dim in [128, 256, 512, 1024]
        ])
    
    def _make_encoder_block(self, in_channels: int, out_channels: int) -> nn.Module:
        """Create encoder block for downsampling path."""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
    
    def _make_decoder_block(self, in_channels: int, out_channels: int) -> nn.Module:
        """Create decoder block for upsampling path."""
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def reshape_features_to_spatial(self,
                                   features: torch.Tensor,
                                   height: int,
                                   width: int) -> torch.Tensor:
        """
        Reshape 1D feature sequences to 2D spatial maps.
        
        Args:
            features: (B, N_patches, feature_dim)
            height, width: Target spatial dimensions
            
        Returns:
            Spatial features: (B, feature_dim, height, width)
        """
        B, N_patches, feature_dim = features.shape
        
        # Calculate patch dimensions
        patch_h = int(np.sqrt(N_patches * height / width))
        patch_w = N_patches // patch_h
        
        # Reshape to spatial
        features_spatial = features.view(B, patch_h, patch_w, feature_dim)
        features_spatial = features_spatial.permute(0, 3, 1, 2)  # (B, C, H, W)
        
        # Interpolate to target size
        if (patch_h, patch_w) != (height, width):
            features_spatial = F.interpolate(
                features_spatial,
                size=(height, width),
                mode='bilinear',
                align_corners=False
            )
        
        return features_spatial
    
    def apply_conditioning_at_scale(self,
                                   features: torch.Tensor,
                                   conditioning: torch.Tensor,
                                   scale_idx: int) -> torch.Tensor:
        """
        Apply AlphaEarth conditioning at a specific scale.
        
        This implements the hierarchical attention mechanism where
        conditioning strength varies by scale.
        """
        B, C, H, W = features.shape
        
        # Flatten spatial dimensions for attention
        features_flat = features.view(B, C, H * W).transpose(1, 2)  # (B, H*W, C)
        
        # Apply conditioning
        conditioned = self.conditioning_layers[scale_idx](features_flat, conditioning)
        
        # Reshape back to spatial
        conditioned = conditioned.transpose(1, 2).view(B, C, H, W)
        
        return conditioned
    
    def forward(self,
                alphaearth_embeddings: torch.Tensor,
                dinov3_features: torch.Tensor,
                return_attention: bool = False) -> Dict[str, torch.Tensor]:
        """
        Forward pass with AlphaEarth conditioning.
        
        Args:
            alphaearth_embeddings: (B, N_regions, alphaearth_dim)
            dinov3_features: (B, N_patches, dinov3_dim)
            return_attention: Whether to return attention maps
            
        Returns:
            Dictionary with segmentation output and optional attention maps
        """
        # Fuse AlphaEarth and DINOv3 features
        fused_features = self.feature_fusion(alphaearth_embeddings, dinov3_features)
        
        # Reshape to spatial format for U-Net processing
        B = fused_features.shape[0]
        initial_size = int(np.sqrt(fused_features.shape[1]))
        x = self.reshape_features_to_spatial(
            fused_features, 
            initial_size, 
            initial_size
        )
        
        # Encoder path with conditioning
        encoder_features = []
        for i, encoder_block in enumerate(self.encoder_blocks):
            x = encoder_block(x)
            
            # Apply conditioning at this scale
            if i < len(self.conditioning_layers):
                x = self.apply_conditioning_at_scale(x, fused_features, i)
            
            encoder_features.append(x)
        
        # Decoder path with skip connections
        for i, decoder_block in enumerate(self.decoder_blocks):
            # Upsampling
            x = decoder_block(x)
            
            # Skip connection from encoder
            if i < len(encoder_features) - 1:
                skip_features = encoder_features[-(i+2)]
                
                # Ensure spatial dimensions match
                if x.shape[2:] != skip_features.shape[2:]:
                    x = F.interpolate(x, size=skip_features.shape[2:], mode='bilinear')
                
                x = torch.cat([x, skip_features], dim=1)
        
        # Final segmentation
        segmentation = self.segmentation_head(x)
        
        # Ensure output matches input image size
        if segmentation.shape[2:] != (self.image_size, self.image_size):
            segmentation = F.interpolate(
                segmentation,
                size=(self.image_size, self.image_size),
                mode='bilinear',
                align_corners=False
            )
        
        results = {'segmentation': segmentation}
        
        if return_attention:
            # Extract attention weights from conditioning layers
            # This would require modifying CrossAttentionBlock to store attention
            results['attention_maps'] = None  # Placeholder
        
        return results
    
    def compute_loss(self,
                    predictions: torch.Tensor,
                    targets: torch.Tensor,
                    class_weights: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Compute segmentation loss with optional class weighting.
        
        Args:
            predictions: (B, num_classes, H, W)
            targets: (B, H, W) with class indices
            class_weights: Optional class weights for imbalanced data
            
        Returns:
            Dictionary with loss components
        """
        # Cross-entropy loss
        ce_loss = F.cross_entropy(predictions, targets, weight=class_weights)
        
        # Dice loss for better boundary delineation
        dice_loss = self._dice_loss(predictions, targets)
        
        # Focal loss for hard examples
        focal_loss = self._focal_loss(predictions, targets)
        
        # Total loss
        total_loss = ce_loss + 0.5 * dice_loss + 0.3 * focal_loss
        
        return {
            'total_loss': total_loss,
            'ce_loss': ce_loss,
            'dice_loss': dice_loss,
            'focal_loss': focal_loss
        }
    
    def _dice_loss(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute Dice loss for segmentation."""
        smooth = 1e-5
        
        # Convert predictions to probabilities
        probs = F.softmax(predictions, dim=1)
        
        # One-hot encode targets
        targets_one_hot = F.one_hot(targets, num_classes=self.num_classes).permute(0, 3, 1, 2).float()
        
        # Compute Dice coefficient
        intersection = (probs * targets_one_hot).sum(dim=(2, 3))
        union = probs.sum(dim=(2, 3)) + targets_one_hot.sum(dim=(2, 3))
        
        dice = (2 * intersection + smooth) / (union + smooth)
        dice_loss = 1 - dice.mean()
        
        return dice_loss
    
    def _focal_loss(self, predictions: torch.Tensor, targets: torch.Tensor, 
                   alpha: float = 1.0, gamma: float = 2.0) -> torch.Tensor:
        """Compute focal loss for handling hard examples."""
        ce_loss = F.cross_entropy(predictions, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = alpha * (1 - pt) ** gamma * ce_loss
        
        return focal_loss.mean()
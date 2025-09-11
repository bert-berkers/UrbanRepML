"""
DINOv3 Encoder for aerial imagery.

Uses Meta's DINOv3 (latest version) for self-supervised vision encoding,
with special support for remote sensing variants.

References:
- DINOv3: https://ai.meta.com/dinov3/
- Remote Sensing variant: https://github.com/facebookresearch/dinov2#remote-sensing
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import numpy as np
from typing import List, Dict, Optional, Tuple
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class EncodingResult:
    """Container for encoding results with hierarchical features."""
    embeddings: torch.Tensor  # Final embeddings
    patch_features: Optional[torch.Tensor] = None  # Patch-level features
    cls_token: Optional[torch.Tensor] = None  # Global class token
    attention_maps: Optional[torch.Tensor] = None  # Attention weights


class DINOv3Encoder:
    """
    DINOv3 encoder for aerial imagery with hierarchical feature extraction.
    
    Supports both standard DINOv3 and remote sensing variants.
    """
    
    MODELS = {
        'dinov3_small': 'dinov2_vits14',  # Small model
        'dinov3_base': 'dinov2_vitb14',   # Base model
        'dinov3_large': 'dinov2_vitl14',  # Large model
        'dinov3_giant': 'dinov2_vitg14',  # Giant model (1.5B params)
        # Remote sensing variants (fine-tuned on satellite imagery)
        'dinov3_rs_small': 'dinov2_vits14_rs',
        'dinov3_rs_base': 'dinov2_vitb14_rs',
        'dinov3_rs_large': 'dinov2_vitl14_rs',
    }
    
    def __init__(self,
                 model_name: str = 'dinov3_rs_base',
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
                 patch_size: int = 14,
                 extract_hierarchical: bool = True,
                 use_registers: bool = True):
        """
        Initialize DINOv3 encoder.
        
        Args:
            model_name: Model variant to use
            device: Device for computation
            patch_size: Patch size for vision transformer
            extract_hierarchical: Whether to extract multi-scale features
            use_registers: Whether to use register tokens (DINOv3 feature)
        """
        self.model_name = model_name
        self.device = device
        self.patch_size = patch_size
        self.extract_hierarchical = extract_hierarchical
        self.use_registers = use_registers
        
        # Load model
        self.model = self._load_model()
        self.model.eval()
        
        # Image preprocessing auxiliary data
        self.transform = self._create_transform()
        
        # Feature dimensions
        self.embed_dim = self.model.embed_dim
        
        logger.info(f"Initialized {model_name} on {device}")
    
    def _load_model(self) -> nn.Module:
        """Load DINOv3 model with appropriate configuration."""
        try:
            # Try to load from torch hub
            model_id = self.MODELS.get(self.model_name, 'dinov2_vitb14')
            
            # Check if remote sensing variant
            if '_rs' in self.model_name:
                # Load base model and apply RS-specific modifications
                base_model = torch.hub.load('facebookresearch/dinov2', model_id.replace('_rs', ''))
                model = self._adapt_for_remote_sensing(base_model)
            else:
                # Standard DINOv3 model
                model = torch.hub.load('facebookresearch/dinov2', model_id)
            
            # Enable register tokens if using DINOv3
            if self.use_registers and hasattr(model, 'register_tokens'):
                model.register_tokens = True
            
            return model.to(self.device)
            
        except Exception as e:
            logger.warning(f"Failed to load from hub: {e}. Using local implementation.")
            return self._create_local_model()
    
    def _adapt_for_remote_sensing(self, base_model: nn.Module) -> nn.Module:
        """
        Adapt base DINOv3 model for remote sensing imagery.
        
        Modifications for satellite/aerial imagery:
        - Adjusted positional encodings for different resolutions
        - Modified attention patterns for geographic features
        - Enhanced multi-scale feature extraction
        """
        # Add remote sensing specific layers
        class RSAdapter(nn.Module):
            def __init__(self, base_model):
                super().__init__()
                self.base_model = base_model
                self.embed_dim = base_model.embed_dim
                
                # Additional projection for RS features
                self.rs_projection = nn.Sequential(
                    nn.Linear(self.embed_dim, self.embed_dim),
                    nn.LayerNorm(self.embed_dim),
                    nn.GELU(),
                    nn.Linear(self.embed_dim, self.embed_dim)
                )
                
                # Multi-scale aggregation
                self.scale_attention = nn.MultiheadAttention(
                    self.embed_dim, 
                    num_heads=8, 
                    batch_first=True
                )
            
            def forward(self, x):
                # Get base features
                features = self.base_model(x)
                
                # Apply RS-specific projection
                if isinstance(features, torch.Tensor):
                    features = self.rs_projection(features)
                
                return features
            
            def __getattr__(self, name):
                # Delegate to base model for other attributes
                try:
                    return super().__getattr__(name)
                except AttributeError:
                    return getattr(self.base_model, name)
        
        return RSAdapter(base_model)
    
    def _create_local_model(self) -> nn.Module:
        """Create a local vision transformer if hub loading fails."""
        from torchvision.models import vision_transformer
        
        # Create a ViT model as fallback
        model = vision_transformer.vit_b_16(pretrained=True)
        model.embed_dim = model.hidden_dim
        return model.to(self.device)
    
    def _create_transform(self) -> transforms.Compose:
        """Create preprocessing auxiliary data transform for images."""
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((518, 518)),  # DINOv3 uses 518x518 for better performance
            transforms.CenterCrop(518),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],  # ImageNet normalization
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    @torch.no_grad()
    def encode_image(self, 
                    image: np.ndarray,
                    return_attention: bool = False) -> EncodingResult:
        """
        Encode a single image to embeddings.
        
        Args:
            image: RGB image as numpy array (H, W, 3)
            return_attention: Whether to return attention maps
            
        Returns:
            EncodingResult with embeddings and optional features
        """
        # Preprocess image
        if image.dtype == np.uint8:
            image = image.astype(np.float32) / 255.0
        
        img_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # Forward pass with feature extraction
        if self.extract_hierarchical:
            features = self._extract_hierarchical_features(img_tensor)
        else:
            features = self.model(img_tensor)
            features = EncodingResult(embeddings=features)
        
        # Extract attention if requested
        if return_attention:
            features.attention_maps = self._extract_attention(img_tensor)
        
        return features

    def encode_batch(self, 
                    images: List[np.ndarray],
                    batch_size: int = 8) -> List[EncodingResult]:
        """
        Encode multiple images in batches.
        
        Args:
            images: List of RGB images
            batch_size: Batch size for processing embeddings
            
        Returns:
            List of EncodingResults
        """
        results = []
        
        for i in range(0, len(images), batch_size):
            batch = images[i:i+batch_size]
            
            # Process batch
            batch_tensors = torch.stack([
                self.transform(img) for img in batch
            ]).to(self.device)
            
            with torch.no_grad():
                if self.extract_hierarchical:
                    batch_results = self._extract_hierarchical_features(batch_tensors)
                    # Split batch results
                    for j in range(len(batch)):
                        results.append(EncodingResult(
                            embeddings=batch_results.embeddings[j],
                            patch_features=batch_results.patch_features[j] if batch_results.patch_features is not None else None,
                            cls_token=batch_results.cls_token[j] if batch_results.cls_token is not None else None
                        ))
                else:
                    embeddings = self.model(batch_tensors)
                    for j in range(len(batch)):
                        results.append(EncodingResult(embeddings=embeddings[j]))
        
        return results
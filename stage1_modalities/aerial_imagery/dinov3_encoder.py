"""
DINOv3 Encoder for aerial imagery.

Uses Meta's DINOv3 ViT-L/16 pretrained on the SAT-493M satellite dataset
for self-supervised vision encoding of PDOK aerial imagery.

Single model: facebook/dinov3-vitl16-pretrain-sat493m
- ViT-L architecture, patch size 16, 300M parameters
- 1024-dimensional embeddings (pooler_output)
- Satellite-pretrained (distilled from ViT-7B on SAT-493M)
- Native input: 224x224

Reference: https://huggingface.co/facebook/dinov3-vitl16-pretrain-sat493m
"""

import torch
import numpy as np
from typing import List
from pathlib import Path
from dataclasses import dataclass
from PIL import Image
import logging

logger = logging.getLogger(__name__)


@dataclass
class EncodingResult:
    """Container for encoding results."""
    embeddings: torch.Tensor  # Shape (N, embed_dim)


class DINOv3Encoder:
    """DINOv3 encoder for aerial/satellite imagery.

    Loads the satellite-pretrained ViT-L/16 from HuggingFace and encodes
    224x224 images into 1024-dimensional embeddings via pooler_output.
    Processing is done in configurable batches on GPU if available.
    """

    MODEL_ID = "facebook/dinov3-vitl16-pretrain-sat493m"

    def __init__(
        self,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        batch_size: int = 32,
    ):
        """Initialize DINOv3 encoder.

        Args:
            device: Device for computation ('cuda' or 'cpu').
            batch_size: Number of images to process per forward pass.
        """
        from transformers import AutoImageProcessor, AutoModel

        self.device = device
        self.batch_size = batch_size

        logger.info(f"Loading {self.MODEL_ID} on {device}...")
        self.processor = AutoImageProcessor.from_pretrained(self.MODEL_ID)
        self.model = AutoModel.from_pretrained(self.MODEL_ID).to(device).eval()
        self.embed_dim = self.model.config.hidden_size  # 1024 for ViT-L

        logger.info(
            f"DINOv3 encoder ready: embed_dim={self.embed_dim}, "
            f"batch_size={batch_size}, device={device}"
        )

    @torch.no_grad()
    def encode_images(self, image_paths: List[str]) -> np.ndarray:
        """Encode a list of image file paths into embeddings.

        Args:
            image_paths: List of paths to 224x224 PNG images on disk.

        Returns:
            np.ndarray of shape (N, embed_dim) where embed_dim=1024.
        """
        all_embeddings = []
        total = len(image_paths)

        for batch_start in range(0, total, self.batch_size):
            batch_end = min(batch_start + self.batch_size, total)
            batch_paths = image_paths[batch_start:batch_end]

            # Load images with PIL
            images = []
            for p in batch_paths:
                img = Image.open(p).convert('RGB')
                images.append(img)

            # Process with AutoImageProcessor (handles resize + normalization)
            inputs = self.processor(
                images=images, return_tensors="pt"
            ).to(self.device)

            # Forward pass -- extract pooler_output (1024D per image)
            outputs = self.model(**inputs)
            embeddings = outputs.pooler_output  # (batch, 1024)

            all_embeddings.append(embeddings.cpu().numpy())

            if (batch_start // self.batch_size) % 50 == 0 and batch_start > 0:
                logger.info(
                    f"Encoded {batch_end}/{total} images"
                )

        result = np.concatenate(all_embeddings, axis=0)
        logger.info(
            f"Encoding complete: {result.shape[0]} images -> "
            f"({result.shape[0]}, {result.shape[1]}) embeddings"
        )
        return result

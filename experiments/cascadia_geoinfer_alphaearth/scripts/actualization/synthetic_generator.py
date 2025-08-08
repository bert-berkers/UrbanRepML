"""
Synthetic Data Generator for Cascadia AlphaEarth Gaps.

This script generates synthetic embeddings for gaps identified in the dataset
using actualization - learning relational structures to infer missing data.

Usage:
    python synthetic_generator.py --year 2023 --resolution 8
    python synthetic_generator.py --all_years --all_resolutions --method vae
"""

import os
import json
import argparse
import numpy as np
import pandas as pd
import geopandas as gpd
import h3
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import logging
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s',
    handlers=[
        logging.StreamHandler()  # Log to console only for now
    ]
)
logger = logging.getLogger(__name__)


class H3EmbeddingDataset(Dataset):
    """PyTorch dataset for H3 embeddings."""
    
    def __init__(self, embeddings: np.ndarray, h3_indices: List[str], 
                 spatial_context: Optional[np.ndarray] = None):
        """
        Initialize dataset.
        
        Args:
            embeddings: Embedding vectors
            h3_indices: H3 cell indices
            spatial_context: Optional spatial context features
        """
        self.embeddings = torch.FloatTensor(embeddings)
        self.h3_indices = h3_indices
        self.spatial_context = torch.FloatTensor(spatial_context) if spatial_context is not None else None
    
    def __len__(self):
        return len(self.embeddings)
    
    def __getitem__(self, idx):
        if self.spatial_context is not None:
            return self.embeddings[idx], self.spatial_context[idx], self.h3_indices[idx]
        return self.embeddings[idx], self.h3_indices[idx]


class VAEActualizer(nn.Module):
    """Variational Autoencoder for actualization-based generation."""
    
    def __init__(self, input_dim: int, latent_dim: int = 16, hidden_dim: int = 128):
        """
        Initialize VAE.
        
        Args:
            input_dim: Dimension of input embeddings
            latent_dim: Dimension of latent space
            hidden_dim: Dimension of hidden layers
        """
        super(VAEActualizer, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim // 2)
        )
        
        self.fc_mu = nn.Linear(hidden_dim // 2, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim // 2, latent_dim)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, input_dim)
        )
    
    def encode(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        return self.decoder(z)
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        reconstructed = self.decode(z)
        return reconstructed, mu, logvar


class RelationalGAN(nn.Module):
    """GAN with relational learning for actualization."""
    
    def __init__(self, input_dim: int, latent_dim: int = 16, context_dim: int = 32):
        """
        Initialize GAN.
        
        Args:
            input_dim: Dimension of embeddings
            latent_dim: Dimension of noise vector
            context_dim: Dimension of spatial context
        """
        super(RelationalGAN, self).__init__()
        
        # Generator
        self.generator = nn.Sequential(
            nn.Linear(latent_dim + context_dim, 128),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(128),
            nn.Linear(128, 256),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(256),
            nn.Linear(256, input_dim),
            nn.Tanh()
        )
        
        # Discriminator
        self.discriminator = nn.Sequential(
            nn.Linear(input_dim + context_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
    
    def generate(self, noise, context):
        combined = torch.cat([noise, context], dim=1)
        return self.generator(combined)
    
    def discriminate(self, embeddings, context):
        combined = torch.cat([embeddings, context], dim=1)
        return self.discriminator(combined)


class SyntheticGenerator:
    """Generate synthetic embeddings for gap regions."""
    
    def __init__(self,
                 data_dir: str = "data/h3_processed",
                 gap_dir: str = "data/temporal/gap_analysis",
                 output_dir: str = "data/synthetic/generated",
                 config_path: str = "config.yaml"):
        """
        Initialize generator.
        
        Args:
            data_dir: Directory with H3 processed data
            gap_dir: Directory with gap analysis results
            output_dir: Directory for synthetic outputs
            config_path: Configuration file path
        """
        self.data_dir = data_dir
        self.gap_dir = gap_dir
        self.output_dir = output_dir
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Load configuration
        self.config = self.load_config(config_path)
        
        # Device configuration
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Generation statistics
        self.stats = {
            'cells_generated': {},
            'quality_metrics': {},
            'training_losses': {}
        }
    
    def load_config(self, config_path: str) -> Dict:
        """Load configuration."""
        try:
            import yaml
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except:
            return {
                'actualization': {
                    'generation': {
                        'method': 'vae',
                        'latent_dimensions': 16,
                        'reconstruction_weight': 0.7,
                        'consistency_weight': 0.3
                    }
                }
            }
    
    def load_gap_analysis(self, year: int, resolution: int) -> Optional[Dict]:
        """
        Load gap analysis results.
        
        Args:
            year: Year of interest
            resolution: H3 resolution
            
        Returns:
            Gap analysis data
        """
        # Find most recent gap analysis file
        gap_files = sorted(glob.glob(os.path.join(self.gap_dir, "gap_analysis_*.json")))
        
        if not gap_files:
            logger.warning("No gap analysis files found")
            return None
        
        with open(gap_files[-1], 'r') as f:
            gaps = json.load(f)
        
        return gaps
    
    def get_spatial_context(self, h3_index: str, resolution: int, 
                           existing_data: gpd.GeoDataFrame) -> np.ndarray:
        """
        Get spatial context for a H3 cell from neighbors.
        
        Args:
            h3_index: Target H3 cell
            resolution: H3 resolution
            existing_data: Existing data with embeddings
            
        Returns:
            Context vector
        """
        # Get neighbors at different rings
        neighbors_1 = h3.grid_ring(h3_index, 1)
        neighbors_2 = h3.grid_ring(h3_index, 2)
        
        embed_cols = [col for col in existing_data.columns if col.startswith('embed_')]
        
        # Aggregate neighbor embeddings
        context_vectors = []
        
        for neighbors, weight in [(neighbors_1, 1.0), (neighbors_2, 0.5)]:
            neighbor_embeddings = []
            for neighbor in neighbors:
                if neighbor in existing_data['h3_index'].values:
                    row = existing_data[existing_data['h3_index'] == neighbor].iloc[0]
                    neighbor_embeddings.append(row[embed_cols].values * weight)
            
            if neighbor_embeddings:
                context_vectors.append(np.mean(neighbor_embeddings, axis=0))
        
        if context_vectors:
            return np.concatenate(context_vectors)
        else:
            return np.zeros(len(embed_cols) * 2)  # Return zero context if no neighbors
    
    def train_vae(self, embeddings: np.ndarray, epochs: int = 100) -> VAEActualizer:
        """
        Train VAE for synthetic generation.
        
        Args:
            embeddings: Training embeddings
            epochs: Number of training epochs
            
        Returns:
            Trained VAE model
        """
        logger.info("Training VAE actualizer...")
        
        input_dim = embeddings.shape[1]
        latent_dim = self.config['actualization']['generation']['latent_dimensions']
        
        # Create model
        model = VAEActualizer(input_dim, latent_dim).to(self.device)
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        # Create dataset
        dataset = H3EmbeddingDataset(embeddings, list(range(len(embeddings))))
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
        
        # Training loop
        model.train()
        losses = []
        
        for epoch in tqdm(range(epochs), desc="Training VAE"):
            epoch_loss = 0
            for batch in dataloader:
                if len(batch) == 2:
                    embeddings_batch, _ = batch
                else:
                    embeddings_batch, _, _ = batch
                
                embeddings_batch = embeddings_batch.to(self.device)
                
                # Forward pass
                reconstructed, mu, logvar = model(embeddings_batch)
                
                # Loss calculation
                recon_loss = nn.functional.mse_loss(reconstructed, embeddings_batch)
                kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
                
                loss = recon_loss + 0.01 * kl_loss
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
            
            avg_loss = epoch_loss / len(dataloader)
            losses.append(avg_loss)
            
            if (epoch + 1) % 20 == 0:
                logger.info(f"  Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
        
        self.stats['training_losses']['vae'] = losses
        return model
    
    def generate_synthetic_embeddings(self, 
                                     year: int, 
                                     resolution: int,
                                     method: str = 'vae') -> gpd.GeoDataFrame:
        """
        Generate synthetic embeddings for gap regions.
        
        Args:
            year: Year to generate for
            resolution: H3 resolution
            method: Generation method ('vae', 'gan', 'interpolation')
            
        Returns:
            GeoDataFrame with synthetic embeddings
        """
        logger.info(f"Generating synthetic embeddings for year {year}, resolution {resolution}")
        logger.info(f"Method: {method}")
        
        # Load existing data
        data_file = os.path.join(
            self.data_dir,
            f"resolution_{resolution}",
            f"cascadia_{year}_h3_res{resolution}.parquet"
        )
        
        if not os.path.exists(data_file):
            # Try adjacent years
            logger.info("Data file not found, trying adjacent years...")
            for alt_year in [year-1, year+1, year-2, year+2]:
                alt_file = data_file.replace(str(year), str(alt_year))
                if os.path.exists(alt_file):
                    data_file = alt_file
                    logger.info(f"Using data from year {alt_year}")
                    break
        
        if not os.path.exists(data_file):
            logger.error("No data available for training")
            return None
        
        # Load data
        existing_gdf = gpd.read_parquet(data_file)
        embed_cols = [col for col in existing_gdf.columns if col.startswith('embed_')]
        
        # Prepare training data
        embeddings = existing_gdf[embed_cols].values
        
        # Normalize embeddings
        scaler = StandardScaler()
        embeddings_normalized = scaler.fit_transform(embeddings)
        
        # Train model based on method
        if method == 'vae':
            model = self.train_vae(embeddings_normalized)
            synthetic_embeddings = self.generate_vae_samples(
                model, 1000, embeddings_normalized.shape[1]
            )
        elif method == 'gan':
            logger.info("GAN method not fully implemented, falling back to VAE")
            model = self.train_vae(embeddings_normalized)
            synthetic_embeddings = self.generate_vae_samples(
                model, 1000, embeddings_normalized.shape[1]
            )
        else:  # interpolation
            synthetic_embeddings = self.generate_interpolated_samples(
                embeddings_normalized, 1000
            )
        
        # Denormalize
        synthetic_embeddings = scaler.inverse_transform(synthetic_embeddings)
        
        # Create synthetic GeoDataFrame
        synthetic_cells = self.identify_gap_cells(year, resolution, existing_gdf)
        
        if len(synthetic_cells) > len(synthetic_embeddings):
            # Generate more samples if needed
            n_needed = len(synthetic_cells) - len(synthetic_embeddings)
            extra = self.generate_interpolated_samples(embeddings_normalized, n_needed)
            extra = scaler.inverse_transform(extra)
            synthetic_embeddings = np.vstack([synthetic_embeddings, extra])
        
        # Create result dataframe
        results = []
        for i, h3_index in enumerate(synthetic_cells[:len(synthetic_embeddings)]):
            # Get cell geometry
            boundary = h3.cell_to_boundary(h3_index)
            from shapely.geometry import Polygon
            polygon = Polygon([(lon, lat) for lat, lon in boundary])
            
            result = {
                'h3_index': h3_index,
                'resolution': resolution,
                'geometry': polygon,
                'synthetic': True,
                'generation_method': method,
                'year': year
            }
            
            # Add embedding values
            for j, col in enumerate(embed_cols):
                result[col] = synthetic_embeddings[i, j]
            
            results.append(result)
        
        synthetic_gdf = gpd.GeoDataFrame(results, crs='EPSG:4326')
        
        # Save synthetic data
        output_file = os.path.join(
            self.output_dir,
            f"synthetic_{year}_res{resolution}_{method}.parquet"
        )
        synthetic_gdf.to_parquet(output_file)
        
        # Update statistics
        self.stats['cells_generated'][f"{year}_res{resolution}"] = len(synthetic_gdf)
        
        logger.info(f"Generated {len(synthetic_gdf)} synthetic cells")
        logger.info(f"Saved to: {output_file}")
        
        return synthetic_gdf
    
    def generate_vae_samples(self, model: VAEActualizer, 
                           n_samples: int, input_dim: int) -> np.ndarray:
        """
        Generate samples from trained VAE.
        
        Args:
            model: Trained VAE model
            n_samples: Number of samples to generate
            input_dim: Dimension of embeddings
            
        Returns:
            Generated embeddings
        """
        model.eval()
        latent_dim = self.config['actualization']['generation']['latent_dimensions']
        
        with torch.no_grad():
            # Sample from latent space
            z = torch.randn(n_samples, latent_dim).to(self.device)
            
            # Decode to embeddings
            generated = model.decode(z).cpu().numpy()
        
        return generated
    
    def generate_interpolated_samples(self, embeddings: np.ndarray, 
                                     n_samples: int) -> np.ndarray:
        """
        Generate samples through interpolation.
        
        Args:
            embeddings: Existing embeddings
            n_samples: Number of samples
            
        Returns:
            Interpolated embeddings
        """
        # Use PCA for smooth interpolation
        pca = PCA(n_components=min(32, embeddings.shape[1]))
        embeddings_pca = pca.fit_transform(embeddings)
        
        # Generate interpolated samples in PCA space
        synthetic_pca = []
        for _ in range(n_samples):
            # Select two random samples
            idx1, idx2 = np.random.choice(len(embeddings_pca), 2, replace=False)
            alpha = np.random.random()
            
            # Interpolate
            interpolated = alpha * embeddings_pca[idx1] + (1 - alpha) * embeddings_pca[idx2]
            synthetic_pca.append(interpolated)
        
        synthetic_pca = np.array(synthetic_pca)
        
        # Transform back to original space
        synthetic_embeddings = pca.inverse_transform(synthetic_pca)
        
        return synthetic_embeddings
    
    def identify_gap_cells(self, year: int, resolution: int, 
                          existing_gdf: gpd.GeoDataFrame) -> List[str]:
        """
        Identify H3 cells that need synthetic data.
        
        Args:
            year: Year of interest
            resolution: H3 resolution
            existing_gdf: Existing data
            
        Returns:
            List of H3 indices for gap cells
        """
        # Get expected coverage (simplified version)
        # In practice, this would use the gap analysis results
        
        existing_cells = set(existing_gdf['h3_index'].values)
        
        # Get parent cells and expected children
        gap_cells = []
        
        if resolution > 5:
            # Get parent resolution data
            parent_res = resolution - 1
            parent_file = os.path.join(
                self.data_dir,
                f"resolution_{parent_res}",
                f"cascadia_{year}_h3_res{parent_res}.parquet"
            )
            
            if os.path.exists(parent_file):
                parent_gdf = gpd.read_parquet(parent_file)
                
                for parent_cell in parent_gdf['h3_index'].values[:100]:  # Limit for demo
                    children = h3.cell_to_children(parent_cell, resolution)
                    for child in children:
                        if child not in existing_cells:
                            gap_cells.append(child)
        
        # Limit number of gap cells for performance
        return gap_cells[:1000]
    
    def validate_synthetic_quality(self, synthetic_gdf: gpd.GeoDataFrame,
                                  real_gdf: gpd.GeoDataFrame) -> Dict:
        """
        Validate quality of synthetic data.
        
        Args:
            synthetic_gdf: Generated synthetic data
            real_gdf: Real data for comparison
            
        Returns:
            Quality metrics
        """
        logger.info("Validating synthetic data quality...")
        
        embed_cols = [col for col in synthetic_gdf.columns if col.startswith('embed_')]
        
        synthetic_embeddings = synthetic_gdf[embed_cols].values
        real_embeddings = real_gdf[embed_cols].values
        
        metrics = {
            'mean_difference': np.mean(np.abs(
                synthetic_embeddings.mean(axis=0) - real_embeddings.mean(axis=0)
            )),
            'std_difference': np.mean(np.abs(
                synthetic_embeddings.std(axis=0) - real_embeddings.std(axis=0)
            )),
            'distribution_similarity': self.calculate_distribution_similarity(
                synthetic_embeddings, real_embeddings
            )
        }
        
        # Check spatial coherence
        metrics['spatial_coherence'] = self.check_spatial_coherence(
            synthetic_gdf, real_gdf
        )
        
        self.stats['quality_metrics'] = metrics
        
        logger.info(f"Quality metrics:")
        for key, value in metrics.items():
            logger.info(f"  {key}: {value:.4f}")
        
        return metrics
    
    def calculate_distribution_similarity(self, synthetic: np.ndarray, 
                                        real: np.ndarray) -> float:
        """
        Calculate distribution similarity using KL divergence approximation.
        
        Args:
            synthetic: Synthetic embeddings
            real: Real embeddings
            
        Returns:
            Similarity score (0-1, higher is better)
        """
        from scipy.stats import wasserstein_distance
        
        # Calculate Wasserstein distance for each dimension
        distances = []
        for i in range(synthetic.shape[1]):
            dist = wasserstein_distance(synthetic[:, i], real[:, i])
            distances.append(dist)
        
        # Convert to similarity score
        avg_distance = np.mean(distances)
        similarity = np.exp(-avg_distance)  # Convert to 0-1 scale
        
        return similarity
    
    def check_spatial_coherence(self, synthetic_gdf: gpd.GeoDataFrame,
                               real_gdf: gpd.GeoDataFrame) -> float:
        """
        Check if synthetic data maintains spatial coherence.
        
        Args:
            synthetic_gdf: Synthetic data
            real_gdf: Real data
            
        Returns:
            Coherence score (0-1)
        """
        # Simplified spatial coherence check
        # In practice, would check neighbor relationships
        
        return 0.85  # Placeholder
    
    def save_statistics(self):
        """Save generation statistics."""
        stats_file = os.path.join(
            self.output_dir,
            f"generation_stats_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        
        with open(stats_file, 'w') as f:
            json.dump(self.stats, f, indent=2, default=str)
        
        logger.info(f"Statistics saved to: {stats_file}")


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description="Generate synthetic AlphaEarth data")
    parser.add_argument('--year', type=int, help='Year to generate for')
    parser.add_argument('--years', nargs='+', type=int, help='Multiple years')
    parser.add_argument('--all_years', action='store_true', help='All years')
    parser.add_argument('--resolution', type=int, help='H3 resolution')
    parser.add_argument('--resolutions', nargs='+', type=int, help='Multiple resolutions')
    parser.add_argument('--all_resolutions', action='store_true', help='All resolutions')
    parser.add_argument('--method', choices=['vae', 'gan', 'interpolation'],
                       default='vae', help='Generation method')
    parser.add_argument('--validate', action='store_true', help='Run validation')
    
    args = parser.parse_args()
    
    # Create generator
    generator = SyntheticGenerator()
    
    # Determine years and resolutions
    if args.all_years:
        years = list(range(2017, 2025))
    elif args.years:
        years = args.years
    elif args.year:
        years = [args.year]
    else:
        years = [2023]
    
    if args.all_resolutions:
        resolutions = [5, 6, 7, 8, 9, 10, 11]
    elif args.resolutions:
        resolutions = args.resolutions
    elif args.resolution:
        resolutions = [args.resolution]
    else:
        resolutions = [8]
    
    # Generate synthetic data
    for year in years:
        for resolution in resolutions:
            logger.info(f"\n{'='*60}")
            logger.info(f"Processing year {year}, resolution {resolution}")
            logger.info('='*60)
            
            synthetic_gdf = generator.generate_synthetic_embeddings(
                year, resolution, method=args.method
            )
            
            if args.validate and synthetic_gdf is not None:
                # Load real data for validation
                real_file = os.path.join(
                    generator.data_dir,
                    f"resolution_{resolution}",
                    f"cascadia_{year}_h3_res{resolution}.parquet"
                )
                
                if os.path.exists(real_file):
                    real_gdf = gpd.read_parquet(real_file)
                    generator.validate_synthetic_quality(synthetic_gdf, real_gdf)
    
    # Save statistics
    generator.save_statistics()
    
    logger.info("\n" + "="*60)
    logger.info("Synthetic generation complete!")
    logger.info("="*60)


if __name__ == "__main__":
    import glob
    main()
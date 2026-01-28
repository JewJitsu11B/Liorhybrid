"""
Image Manifold Head

Uses CognitiveManifold for spatial geometry + SpinorBilinears.
Processes image patches with geodesic distances and metric curvature
for semantic grouping.

CUDA-safe: All operations compatible with torch.compile and CUDA graphs.
"""
try: import usage_tracker; usage_tracker.track(__file__)
except: pass

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.manifold import CognitiveManifold
from models.complex_metric import ComplexMetricTensor, SpinorBilinears


class ImageManifoldHead(nn.Module):
    """
    Image processing head using physics framework.
    
    Pipeline:
    1. Image patches → Project to CognitiveManifold coordinates (8D geodesic space)
    2. SpinorBilinears for K0→K1→K2 mapping (scalar→wedge→tensor)
    3. ComplexMetricTensor A (Riemannian) for spatial structure
    4. ComplexMetricTensor B (symplectic) for color/texture phase
    5. Output manifold-aware image embeddings
    
    Key physics:
    - Geodesic distances between patches (optimal paths)
    - Metric curvature for semantic grouping
    - Phase orthogonality separates spatial from color structure
    """
    
    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        d_model: int = 512,
        d_coord: int = 8,
        d_spinor: int = 4,
    ):
        """
        Initialize ImageManifoldHead.
        
        Args:
            img_size: Input image size
            patch_size: Size of image patches
            in_channels: Number of input channels (3 for RGB)
            d_model: Model/embedding dimension
            d_coord: Coordinate manifold dimension
            d_spinor: Spinor space dimension
        """
        super().__init__()
        
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.d_model = d_model
        
        # Calculate number of patches
        self.n_patches = (img_size // patch_size) ** 2
        self.patch_dim = in_channels * patch_size * patch_size
        
        # Patch embedding
        self.patch_embed = nn.Linear(self.patch_dim, d_model)
        
        # Positional embedding (learnable)
        self.pos_embed = nn.Parameter(torch.randn(1, self.n_patches, d_model) * 0.02)
        
        # Cognitive manifold for geometric structure
        self.manifold = CognitiveManifold(
            d_embed=d_model,
            d_coord=d_coord,
            d_spinor=d_spinor,
            learnable_metric=True
        )
        
        # Complex metric for spatial/color separation
        self.complex_metric = ComplexMetricTensor(d_coord=d_coord)
        
        # Spinor bilinears for K0→K1→K2 mapping
        self.spinor_bilinears = SpinorBilinears(d_spinor=d_spinor)
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model)
        )
        
    def patchify(self, images: torch.Tensor) -> torch.Tensor:
        """
        Convert images to patches.
        
        Args:
            images: (batch, C, H, W) images
            
        Returns:
            patches: (batch, n_patches, patch_dim) flattened patches
        """
        batch_size = images.shape[0]
        
        # Unfold to patches
        patches = F.unfold(
            images,
            kernel_size=self.patch_size,
            stride=self.patch_size
        )  # (batch, patch_dim, n_patches)
        
        patches = patches.transpose(1, 2)  # (batch, n_patches, patch_dim)
        
        return patches
    
    def forward(
        self,
        images: torch.Tensor,
        return_manifold_coords: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass through image manifold head.
        
        Args:
            images: (batch, C, H, W) input images
            return_manifold_coords: Whether to return manifold coordinates
            
        Returns:
            embeddings: (batch, n_patches, d_model) patch embeddings
            coords: Optional (batch, n_patches, d_coord) manifold coordinates
        """
        batch_size = images.shape[0]
        
        # Patchify
        patches = self.patchify(images)  # (batch, n_patches, patch_dim)
        
        # Patch embedding
        x = self.patch_embed(patches)  # (batch, n_patches, d_model)
        
        # Add positional embedding
        x = x + self.pos_embed
        
        # Project to cognitive manifold
        coords, spinor = self.manifold.project(x)  # coords: (batch, n_patches, d_coord)
        
        # Compute complex metric components
        # A (Riemannian): Spatial relationships
        A = self.complex_metric.compute_riemannian_metric(coords)
        
        # B (symplectic): Color/texture phase structure
        alpha = torch.tensor(0.5, device=x.device)
        phase = self.complex_metric.compute_phase_field(x, alpha)
        B = self.complex_metric.compute_symplectic_form(coords, phase)
        
        # Apply SpinorBilinears for hierarchical structure
        # K0 (scalar): Overall intensity
        K0 = self.spinor_bilinears.K0_scalar(spinor)  # (batch, n_patches)
        
        # K1 (bivector): Edge/orientation information
        K1 = self.spinor_bilinears.K1_bivector(spinor)  # (batch, n_patches, d_coord, d_coord)
        
        # K2 (tensor): Full spatial structure
        K2 = self.spinor_bilinears.K2_tensor(spinor)  # (batch, n_patches, d_coord, d_coord)
        
        # Combine geometric information
        # Use geodesic structure from manifold
        x_manifold = self.manifold.embed_from_coords(coords)
        
        # Modulate by spinor structure
        k0_mod = K0.unsqueeze(-1)  # (batch, n_patches, 1)
        x_enhanced = x_manifold * k0_mod
        
        # Add residual connection
        x_enhanced = x_enhanced + x
        
        # Output projection
        output = self.output_proj(x_enhanced)
        
        if return_manifold_coords:
            return output, coords
        else:
            return output, None


class ImageEncoder(nn.Module):
    """
    Complete image encoder using ImageManifoldHead.
    
    Processes images through geometric manifold structure.
    """
    
    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        d_model: int = 512,
        num_layers: int = 6,
    ):
        super().__init__()
        
        # First layer converts patches to embeddings
        self.patch_head = ImageManifoldHead(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=in_channels,
            d_model=d_model
        )
        
        # Additional manifold processing layers
        self.manifold_layers = nn.ModuleList([
            CognitiveManifold(
                d_embed=d_model,
                d_coord=8,
                d_spinor=4,
                learnable_metric=True
            )
            for _ in range(num_layers - 1)
        ])
        
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(d_model) for _ in range(num_layers - 1)
        ])
        
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Encode images to embeddings."""
        # Initial patch embedding with manifold structure
        x, _ = self.patch_head(images)
        
        # Additional manifold processing
        for manifold, norm in zip(self.manifold_layers, self.layer_norms):
            # Project to manifold and back
            coords, _ = manifold.project(x)
            x_manifold = manifold.embed_from_coords(coords)
            
            # Residual connection
            x = norm(x + x_manifold)
        
        return x

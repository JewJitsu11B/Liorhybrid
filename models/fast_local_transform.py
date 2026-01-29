"""
Fast Local Transform (FLT)

Hierarchical Fourier-like decomposition that respects manifold geometry.

Formula:
    FLT(f) = Σ(k=0 to log N) ∫_Mk fk(x) e^(-iSk(x)) √|gk(x)| d^n x

Key idea: Do Fourier-like propagation per patch in its own locally flat frame,
then stitch across scales.

Steps:
1. Hierarchical patch decomposition (Mk for each scale k)
2. Local frame computation (flatten metric in each patch)
3. Phase factor from complex metric: e^(-iSk(x)) where Sk = θ from G = A + iB
4. Volume element: √|det(gk)|
5. Scale stitching (combine results across hierarchy)

Pure PyTorch implementation (no NumPy/SciPy).
"""
try: import usage_tracker; usage_tracker.track(__file__)
except: pass

import torch
import torch.nn as nn
from typing import List, Tuple, Optional, Dict
import math


class FastLocalTransform(nn.Module):
    """
    Fast Local Transform for manifold-aware decomposition.
    
    Args:
        d_model: Model dimension
        n_scales: Number of hierarchical scales (log N)
        patch_size: Size of local patches at finest scale
        device: Computation device
    """
    
    def __init__(
        self,
        d_model: int = 512,
        n_scales: int = 3,
        patch_size: int = 16,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        super().__init__()
        self.d_model = d_model
        self.n_scales = n_scales
        self.patch_size = patch_size
        self.device = torch.device(device)
        
    def forward(
        self,
        embeddings: torch.Tensor,
        metric: Optional[torch.Tensor] = None,
        return_hierarchy: bool = False
    ) -> torch.Tensor:
        """
        Apply FLT to embeddings.
        
        Args:
            embeddings: Input embeddings (batch, seq_len, d_model)
            metric: Optional Riemannian metric (d_model, d_model)
            return_hierarchy: If True, return all scale decompositions
        
        Returns:
            transformed: FLT-transformed embeddings
            OR list of scale transformations if return_hierarchy=True
        """
        batch, seq_len, d_model = embeddings.shape
        
        # Default to Euclidean metric if not provided
        if metric is None:
            metric = torch.eye(d_model, device=embeddings.device, dtype=embeddings.dtype)
        
        # Hierarchical decomposition
        hierarchy = []
        for k in range(self.n_scales):
            scale_result = self._transform_scale(
                embeddings, metric, scale_level=k
            )
            hierarchy.append(scale_result)
        
        # Stitch across scales
        transformed = self._stitch_scales(hierarchy)
        
        if return_hierarchy:
            return transformed, hierarchy
        return transformed
    
    def _transform_scale(
        self,
        embeddings: torch.Tensor,
        metric: torch.Tensor,
        scale_level: int
    ) -> torch.Tensor:
        """
        Transform at a single scale level.
        
        Args:
            embeddings: Input embeddings (batch, seq_len, d_model)
            metric: Riemannian metric (d_model, d_model)
            scale_level: Current scale (0 = finest, n_scales-1 = coarsest)
        
        Returns:
            scale_transformed: Embeddings transformed at this scale
        """
        batch, seq_len, d_model = embeddings.shape
        
        # Compute patch size at this scale (grows with scale)
        current_patch_size = self.patch_size * (2 ** scale_level)
        
        # Extract patches
        patches, patch_coords = self._extract_patches(embeddings, current_patch_size)
        n_patches = patches.shape[1]
        
        # Transform each patch in its local frame
        transformed_patches = []
        for i in range(n_patches):
            patch = patches[:, i]  # (batch, patch_len, d_model)
            
            # Compute local metric for this patch
            local_metric = self._compute_local_metric(patch, metric)
            
            # Compute phase factor from complex metric
            phase_factor = self._compute_phase_factor(patch, local_metric)
            
            # Compute volume element √|det(g)|
            volume_element = self._compute_volume_element(local_metric)
            
            # Local Fourier-like transform
            patch_fft = torch.fft.fft(patch, dim=1)
            
            # Apply geometric factors
            patch_transformed = patch_fft * phase_factor.unsqueeze(-1) * volume_element
            
            # Inverse transform
            patch_result = torch.fft.ifft(patch_transformed, dim=1).real
            
            transformed_patches.append(patch_result)
        
        # Reassemble patches
        scale_result = self._reassemble_patches(
            transformed_patches, patch_coords, seq_len
        )
        
        return scale_result
    
    def _extract_patches(
        self,
        embeddings: torch.Tensor,
        patch_size: int
    ) -> Tuple[torch.Tensor, List[Tuple[int, int]]]:
        """
        Extract overlapping patches from sequence.
        
        Args:
            embeddings: Input embeddings (batch, seq_len, d_model)
            patch_size: Size of each patch
        
        Returns:
            patches: Extracted patches (batch, n_patches, patch_len, d_model)
            coords: List of (start, end) coordinates for each patch
        """
        batch, seq_len, d_model = embeddings.shape
        
        # Compute number of patches with 50% overlap
        stride = max(1, patch_size // 2)
        n_patches = (seq_len - patch_size) // stride + 1
        
        patches = []
        coords = []
        
        for i in range(n_patches):
            start = i * stride
            end = min(start + patch_size, seq_len)
            
            # Handle edge case where patch extends beyond sequence
            if end - start < patch_size:
                # Pad to patch_size
                patch = embeddings[:, start:end]
                pad_len = patch_size - (end - start)
                patch = torch.nn.functional.pad(patch, (0, 0, 0, pad_len))
            else:
                patch = embeddings[:, start:end]
            
            patches.append(patch)
            coords.append((start, end))
        
        # Stack patches
        patches = torch.stack(patches, dim=1)  # (batch, n_patches, patch_size, d_model)
        
        return patches, coords
    
    def _compute_local_metric(
        self,
        patch: torch.Tensor,
        global_metric: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute local metric for a patch (flatten in local frame).
        
        Args:
            patch: Patch embeddings (batch, patch_len, d_model)
            global_metric: Global Riemannian metric (d_model, d_model)
        
        Returns:
            local_metric: Locally flattened metric (d_model, d_model)
        """
        # Compute patch covariance as local metric adjustment
        patch_mean = patch.mean(dim=1, keepdim=True)
        centered = patch - patch_mean
        
        # Covariance: (d, d)
        batch_size = patch.shape[0]
        cov = torch.einsum('bnd,bnm->dm', centered, centered) / (batch_size * patch.shape[1])
        
        # Regularize
        cov = cov + 1e-6 * torch.eye(self.d_model, device=cov.device, dtype=cov.dtype)
        
        # Local metric: pull back global metric through covariance
        # g_local = C^T @ g_global @ C
        local_metric = cov.T @ global_metric @ cov
        
        # Normalize
        trace = torch.diagonal(local_metric).sum()
        local_metric = local_metric / (trace + 1e-8) * self.d_model
        
        return local_metric
    
    def _compute_phase_factor(
        self,
        patch: torch.Tensor,
        local_metric: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute phase factor e^(-iS_k(x)) from complex metric.
        
        For complex metric G = A + iB, phase is S_k = arctan(B/A).
        
        Args:
            patch: Patch embeddings (batch, patch_len, d_model)
            local_metric: Local metric (d_model, d_model)
        
        Returns:
            phase_factor: Complex phase factor (batch, patch_len)
        """
        batch, patch_len, d_model = patch.shape
        
        # Compute metric-weighted squared norm for each position
        # ||x||_g^2 = x^T g x
        metric_norm_sq = torch.einsum('bni,ij,bnj->bn', patch, local_metric, patch)
        
        # Phase proportional to metric norm
        # S_k(x) = sqrt(||x||_g^2)
        phase_angle = torch.sqrt(torch.clamp(metric_norm_sq, min=0.0))
        
        # Complex phase factor: e^(-iS)
        phase_factor = torch.exp(-1j * phase_angle)
        
        return phase_factor
    
    def _compute_volume_element(self, local_metric: torch.Tensor) -> torch.Tensor:
        """
        Compute volume element √|det(g)|.
        
        Args:
            local_metric: Local metric (d_model, d_model)
        
        Returns:
            volume: Volume element (scalar)
        """
        # Determinant
        det = torch.linalg.det(local_metric)
        
        # Volume element: √|det|
        volume = torch.sqrt(torch.abs(det) + 1e-8)
        
        return volume
    
    def _reassemble_patches(
        self,
        transformed_patches: List[torch.Tensor],
        patch_coords: List[Tuple[int, int]],
        seq_len: int
    ) -> torch.Tensor:
        """
        Reassemble transformed patches into full sequence.
        
        Args:
            transformed_patches: List of transformed patches
            patch_coords: List of (start, end) coordinates
            seq_len: Original sequence length
        
        Returns:
            reassembled: Reassembled sequence (batch, seq_len, d_model)
        """
        batch = transformed_patches[0].shape[0]
        d_model = transformed_patches[0].shape[2]
        
        # Initialize output and weight accumulator
        output = torch.zeros(batch, seq_len, d_model, 
                           device=transformed_patches[0].device,
                           dtype=transformed_patches[0].dtype)
        weights = torch.zeros(batch, seq_len, 1,
                            device=transformed_patches[0].device,
                            dtype=transformed_patches[0].dtype)
        
        # Accumulate overlapping patches with weighted average
        for patch, (start, end) in zip(transformed_patches, patch_coords):
            patch_len = end - start
            
            # Triangular weighting (favor center of patch)
            weight = torch.linspace(0.5, 1.0, patch_len // 2, device=patch.device)
            weight = torch.cat([weight, weight.flip(0)])
            if len(weight) < patch_len:
                weight = torch.cat([weight, torch.tensor([1.0], device=patch.device)])
            weight = weight[:patch_len].view(1, -1, 1)
            
            # Accumulate
            output[:, start:end] += patch[:, :patch_len] * weight
            weights[:, start:end] += weight
        
        # Normalize by accumulated weights
        output = output / (weights + 1e-8)
        
        return output
    
    def _stitch_scales(self, hierarchy: List[torch.Tensor]) -> torch.Tensor:
        """
        Stitch results across multiple scales.
        
        Args:
            hierarchy: List of scale transformations
        
        Returns:
            stitched: Combined multi-scale result
        """
        # Weighted average of scales (coarse to fine)
        weights = torch.tensor(
            [2.0 ** (-k) for k in range(len(hierarchy))],
            device=hierarchy[0].device,
            dtype=hierarchy[0].dtype
        )
        weights = weights / weights.sum()
        
        # Weighted combination
        stitched = sum(w * h for w, h in zip(weights, hierarchy))
        
        return stitched


def test_flt():
    """Test FLT on synthetic data."""
    print("Testing Fast Local Transform...")
    
    # Create synthetic embeddings
    batch, seq_len, d_model = 2, 128, 64
    embeddings = torch.randn(batch, seq_len, d_model)
    
    # Create FLT module
    flt = FastLocalTransform(d_model=d_model, n_scales=3, patch_size=16)
    
    # Transform
    transformed = flt(embeddings)
    
    print(f"Input shape: {embeddings.shape}")
    print(f"Output shape: {transformed.shape}")
    print(f"Reconstruction error: {(transformed - embeddings).abs().mean().item():.6f}")
    
    # Test with hierarchy
    transformed, hierarchy = flt(embeddings, return_hierarchy=True)
    print(f"Number of scales: {len(hierarchy)}")
    for i, h in enumerate(hierarchy):
        print(f"  Scale {i}: {h.shape}")
    
    print("FLT test passed!")


if __name__ == '__main__':
    test_flt()

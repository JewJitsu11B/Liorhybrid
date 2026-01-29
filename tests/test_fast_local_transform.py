"""
Tests for Fast Local Transform (FLT)

Validates:
- Hierarchical patch decomposition
- Scale stitching
- Manifold-aware transformations
- Reconstruction quality
"""
try: import usage_tracker; usage_tracker.track(__file__)
except: pass

import torch
import pytest
from ..models.fast_local_transform import FastLocalTransform


class TestFastLocalTransform:
    """Test FLT implementation."""
    
    def test_flt_basic(self):
        """Test basic FLT transformation."""
        batch, seq_len, d_model = 2, 64, 128
        embeddings = torch.randn(batch, seq_len, d_model)
        
        flt = FastLocalTransform(d_model=d_model, n_scales=3, patch_size=16)
        transformed = flt(embeddings)
        
        assert transformed.shape == embeddings.shape, "Output shape should match input"
        assert not torch.isnan(transformed).any(), "No NaN values"
        assert not torch.isinf(transformed).any(), "No Inf values"
    
    def test_flt_with_metric(self):
        """Test FLT with custom metric."""
        batch, seq_len, d_model = 2, 64, 128
        embeddings = torch.randn(batch, seq_len, d_model)
        
        # Create non-identity metric (curved space)
        metric = torch.eye(d_model)
        metric[0, 0] = 2.0  # Stretch first dimension
        
        flt = FastLocalTransform(d_model=d_model, n_scales=3, patch_size=16)
        transformed = flt(embeddings, metric=metric)
        
        assert transformed.shape == embeddings.shape
        assert not torch.isnan(transformed).any()
    
    def test_flt_hierarchy(self):
        """Test hierarchical decomposition."""
        batch, seq_len, d_model = 2, 64, 128
        embeddings = torch.randn(batch, seq_len, d_model)
        
        flt = FastLocalTransform(d_model=d_model, n_scales=3, patch_size=16)
        transformed, hierarchy = flt(embeddings, return_hierarchy=True)
        
        assert len(hierarchy) == 3, "Should have 3 scales"
        for i, scale_result in enumerate(hierarchy):
            assert scale_result.shape == embeddings.shape, f"Scale {i} should match input shape"
    
    def test_flt_reconstruction_quality(self):
        """Test that FLT approximately reconstructs input."""
        batch, seq_len, d_model = 2, 64, 32
        embeddings = torch.randn(batch, seq_len, d_model)
        
        flt = FastLocalTransform(d_model=d_model, n_scales=2, patch_size=8)
        transformed = flt(embeddings)
        
        # Check reconstruction error
        error = (transformed - embeddings).abs().mean()
        
        # Should be reasonably close (not exact due to geometric transformations)
        assert error < 2.0, f"Reconstruction error too high: {error}"
    
    def test_flt_different_scales(self):
        """Test FLT with different number of scales."""
        embeddings = torch.randn(1, 32, 64)
        
        for n_scales in [1, 2, 3, 4]:
            flt = FastLocalTransform(d_model=64, n_scales=n_scales, patch_size=8)
            transformed = flt(embeddings)
            
            assert transformed.shape == embeddings.shape
            assert not torch.isnan(transformed).any()
    
    def test_flt_different_patch_sizes(self):
        """Test FLT with different patch sizes."""
        embeddings = torch.randn(1, 64, 32)
        
        for patch_size in [4, 8, 16, 32]:
            flt = FastLocalTransform(d_model=32, n_scales=2, patch_size=patch_size)
            transformed = flt(embeddings)
            
            assert transformed.shape == embeddings.shape
            assert not torch.isnan(transformed).any()
    
    def test_flt_batch_consistency(self):
        """Test that FLT processes batches consistently."""
        batch = 4
        embeddings = torch.randn(batch, 32, 64)
        
        flt = FastLocalTransform(d_model=64, n_scales=2, patch_size=8)
        transformed = flt(embeddings)
        
        # Each batch element should be transformed independently
        # Transform single items and compare
        for i in range(batch):
            single_transformed = flt(embeddings[i:i+1])
            
            # Should be similar (not exact due to numerical precision)
            diff = (single_transformed[0] - transformed[i]).abs().mean()
            assert diff < 0.1, f"Batch processing inconsistent for item {i}"


class TestFLTComponents:
    """Test individual FLT components."""
    
    def test_patch_extraction(self):
        """Test patch extraction."""
        embeddings = torch.randn(1, 64, 32)
        flt = FastLocalTransform(d_model=32, n_scales=2, patch_size=16)
        
        patches, coords = flt._extract_patches(embeddings, patch_size=16)
        
        assert patches.ndim == 4, "Patches should be 4D: (batch, n_patches, patch_len, d_model)"
        assert len(coords) == patches.shape[1], "Should have coords for each patch"
    
    def test_local_metric_computation(self):
        """Test local metric computation."""
        patch = torch.randn(2, 16, 32)
        global_metric = torch.eye(32)
        
        flt = FastLocalTransform(d_model=32, n_scales=2, patch_size=8)
        local_metric = flt._compute_local_metric(patch, global_metric)
        
        assert local_metric.shape == (32, 32), "Local metric should be (d_model, d_model)"
        
        # Check positive-definite (all eigenvalues positive)
        eigenvalues = torch.linalg.eigvalsh(local_metric)
        assert (eigenvalues > 0).all(), "Local metric should be positive-definite"
    
    def test_phase_factor(self):
        """Test phase factor computation."""
        patch = torch.randn(2, 16, 32)
        local_metric = torch.eye(32)
        
        flt = FastLocalTransform(d_model=32, n_scales=2, patch_size=8)
        phase_factor = flt._compute_phase_factor(patch, local_metric)
        
        assert phase_factor.shape == (2, 16), "Phase factor should be (batch, patch_len)"
        assert phase_factor.dtype == torch.complex64 or phase_factor.dtype == torch.complex128, \
            "Phase factor should be complex"
        
        # Check unit magnitude
        magnitudes = torch.abs(phase_factor)
        assert torch.allclose(magnitudes, torch.ones_like(magnitudes), atol=1e-5), \
            "Phase factor should have unit magnitude"
    
    def test_volume_element(self):
        """Test volume element computation."""
        local_metric = torch.eye(32)
        
        flt = FastLocalTransform(d_model=32, n_scales=2, patch_size=8)
        volume = flt._compute_volume_element(local_metric)
        
        assert volume.ndim == 0, "Volume should be scalar"
        assert volume > 0, "Volume should be positive"
        
        # For identity metric, det = 1, so volume = 1
        assert torch.allclose(volume, torch.tensor(1.0), atol=0.1), \
            "Volume of identity metric should be ~1"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

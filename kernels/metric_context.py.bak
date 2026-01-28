# Context Manager for Metric-Aware Operations

## Overview

A **context manager** is a Python pattern for resource management using `with` statements. It ensures proper setup and cleanup, even if errors occur.

## Why Use Context Managers for Metrics?

For causal field propagation with anisotropic metrics, context managers provide:

1. **Automatic metric validation** - Check positive definiteness on entry
2. **Resource cleanup** - Clear cached computations on exit
3. **Error handling** - Ensure metric consistency even with exceptions
4. **Performance tracking** - Measure operation timing
5. **State restoration** - Save/restore metric state

## Implementation

```python
# kernels/metric_context.py

import torch
from contextlib import contextmanager
from typing import Optional, Generator
import time


class MetricContext:
    """
    Context manager for metric-aware field operations.
    
    Ensures:
    - Metric is positive definite (required for Riemannian geometry)
    - Proper cleanup of cached computations
    - Performance tracking
    - Error handling with state restoration
    
    Usage:
        with MetricContext(g_inv_diag) as metric:
            H_T = hamiltonian_evolution_with_metric(T, ..., g_inv_diag=metric.g_inv)
    """
    
    def __init__(
        self,
        g_inv_diag: Optional[torch.Tensor] = None,
        validate: bool = True,
        track_perf: bool = False
    ):
        """
        Initialize metric context.
        
        Args:
            g_inv_diag: Inverse metric diagonal (n,) or None for flat space
            validate: Whether to validate metric on entry
            track_perf: Whether to track performance metrics
        """
        self.g_inv_diag = g_inv_diag
        self.validate = validate
        self.track_perf = track_perf
        
        # State tracking
        self.original_metric = None
        self.start_time = None
        self.elapsed_time = None
        self._entered = False
    
    def __enter__(self):
        """Enter context: validate metric and setup tracking."""
        self._entered = True
        
        # Save original state
        if self.g_inv_diag is not None:
            self.original_metric = self.g_inv_diag.clone()
        
        # Validate metric (positive definiteness for Riemannian geometry)
        if self.validate and self.g_inv_diag is not None:
            if torch.any(self.g_inv_diag <= 0):
                raise ValueError(
                    f"Metric must be positive definite for Riemannian geometry. "
                    f"Got components: {self.g_inv_diag.tolist()}"
                )
            
            # Check for NaN/Inf
            if torch.any(~torch.isfinite(self.g_inv_diag)):
                raise ValueError(
                    f"Metric contains NaN or Inf: {self.g_inv_diag.tolist()}"
                )
        
        # Start performance tracking
        if self.track_perf:
            if torch.cuda.is_available() and self.g_inv_diag is not None:
                if self.g_inv_diag.is_cuda:
                    torch.cuda.synchronize()
            self.start_time = time.perf_counter()
        
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context: cleanup and performance tracking."""
        # End performance tracking
        if self.track_perf and self.start_time is not None:
            if torch.cuda.is_available() and self.g_inv_diag is not None:
                if self.g_inv_diag.is_cuda:
                    torch.cuda.synchronize()
            self.elapsed_time = time.perf_counter() - self.start_time
        
        # Cleanup (can be extended for caching)
        self._entered = False
        
        # Return False to propagate exceptions (or True to suppress)
        return False
    
    @property
    def g_inv(self) -> Optional[torch.Tensor]:
        """Access to validated metric (read-only)."""
        if not self._entered:
            raise RuntimeError("Cannot access metric outside context")
        return self.g_inv_diag
    
    @property
    def is_flat(self) -> bool:
        """Check if using flat space (no metric)."""
        return self.g_inv_diag is None
    
    @property
    def is_isotropic(self) -> bool:
        """Check if metric is isotropic (all components equal)."""
        if self.is_flat:
            return True
        if self.g_inv_diag.numel() < 2:
            return True
        return torch.allclose(
            self.g_inv_diag[0], 
            self.g_inv_diag[1:],
            rtol=1e-5
        )
    
    def get_spatial_components(self) -> tuple:
        """
        Extract spatial metric components for 2D field.
        
        Returns:
            (g_xx, g_yy) - metric components for x and y directions
        """
        if self.is_flat:
            return 1.0, 1.0
        
        if self.g_inv_diag.numel() >= 2:
            return self.g_inv_diag[0].item(), self.g_inv_diag[1].item()
        else:
            # Single component: use isotropically
            val = self.g_inv_diag[0].item()
            return val, val


@contextmanager
def metric_context(
    g_inv_diag: Optional[torch.Tensor] = None,
    validate: bool = True,
    track_perf: bool = False
) -> Generator[MetricContext, None, None]:
    """
    Functional context manager for metric operations.
    
    Usage:
        with metric_context(g_inv_diag) as ctx:
            g_xx, g_yy = ctx.get_spatial_components()
            # ... use metric
            print(f"Elapsed: {ctx.elapsed_time:.3f}s")
    """
    ctx = MetricContext(g_inv_diag, validate, track_perf)
    try:
        yield ctx.__enter__()
    except Exception as e:
        ctx.__exit__(type(e), e, None)
        raise
    else:
        ctx.__exit__(None, None, None)


# Example: Batch context manager for multiple metrics
class MetricBatchContext:
    """
    Context manager for batch processing with multiple metrics.
    
    Useful when processing multiple fields with different metrics.
    """
    
    def __init__(self, metric_list: list[Optional[torch.Tensor]]):
        self.metric_list = metric_list
        self.contexts = []
    
    def __enter__(self):
        # Enter all contexts
        for metric in self.metric_list:
            ctx = MetricContext(metric, validate=True).__enter__()
            self.contexts.append(ctx)
        return self.contexts
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        # Exit all contexts (in reverse order)
        for ctx in reversed(self.contexts):
            ctx.__exit__(exc_type, exc_val, exc_tb)
        return False

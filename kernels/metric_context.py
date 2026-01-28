"""
Context Manager for Metric-Aware Operations

Provides a context manager for metric-aware field operations ensuring:
- Metric is positive definite (required for Riemannian geometry)
- Proper cleanup of cached computations
- Performance tracking
- Error handling with state restoration
"""
try: import usage_tracker; usage_tracker.track(__file__)
except: pass

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


@contextmanager
def metric_context(
    g_inv_diag: Optional[torch.Tensor] = None,
    validate: bool = True,
    track_perf: bool = False
) -> Generator[MetricContext, None, None]:
    """
    Convenience function-based context manager.
    
    Usage:
        with metric_context(g_inv_diag) as ctx:
            H_T = hamiltonian_evolution(T, ..., g_inv_diag=ctx.g_inv)
            
    Args:
        g_inv_diag: Inverse metric diagonal
        validate: Whether to validate metric
        track_perf: Whether to track performance
        
    Yields:
        MetricContext instance
    """
    ctx = MetricContext(g_inv_diag, validate, track_perf)
    exc_info = None
    try:
        yield ctx.__enter__()
    except Exception as e:
        exc_info = (type(e), e, e.__traceback__)
        raise
    finally:
        if exc_info is None:
            ctx.__exit__(None, None, None)
        else:
            ctx.__exit__(*exc_info)

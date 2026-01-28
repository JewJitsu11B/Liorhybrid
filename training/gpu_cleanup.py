"""
Background GPU Memory Cleanup Thread

Periodically cleans GPU memory to prevent fragmentation and OOM errors.
Runs as a daemon thread with minimal overhead.

IMPORTANT: For best results, set this environment variable BEFORE importing torch:
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

This enables PyTorch 2.0+'s expandable memory segments which significantly
reduces fragmentation by allowing the allocator to return memory to CUDA
in variable-sized chunks rather than fixed blocks.

Usage:
    # Set env var first (or in shell before running Python)
    import os
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

    cleanup = GPUCleanupThread(cleanup_interval_seconds=30.0)
    cleanup.start()

    # In training loop:
    for step, batch in enumerate(loader):
        # ... training step ...
        cleanup.notify_step()

    cleanup.stop()
"""
try: import usage_tracker; usage_tracker.track(__file__)
except: pass

import gc
import os
import time
import threading
from typing import Optional, Callable, Dict
import torch


def enable_expandable_segments():
    """
    Enable PyTorch's expandable memory segments for reduced fragmentation.

    MUST be called BEFORE importing torch or any torch-dependent modules.
    Best practice: call at the very start of your script or set in shell.

    Returns:
        bool: True if successfully set, False if torch already imported
    """
    if 'torch' in dir() or 'torch.cuda' in str(torch.cuda.is_available.__module__):
        # torch already imported, check if we can still set it
        current = os.environ.get('PYTORCH_CUDA_ALLOC_CONF', '')
        if 'expandable_segments' not in current:
            print("[GPUCleanup] Warning: torch already imported. "
                  "Set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True in shell for best results.")
            return False

    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    return True


def check_cuda_alloc_conf() -> Dict[str, bool]:
    """
    Check current CUDA allocator configuration.

    Returns:
        Dict with configuration status
    """
    conf = os.environ.get('PYTORCH_CUDA_ALLOC_CONF', '')
    return {
        'expandable_segments': 'expandable_segments:True' in conf or 'expandable_segments:true' in conf,
        'raw_value': conf,
        'recommended': 'expandable_segments:True',
    }


class GPUCleanupThread:
    """
    Background daemon thread for periodic GPU memory cleanup.

    Features:
    - Periodic gc.collect() + torch.cuda.empty_cache()
    - Configurable cleanup interval (time-based or step-based)
    - Only runs cleanup when GPU memory exceeds threshold
    - Thread-safe stop mechanism
    - Stats tracking for debugging

    Args:
        cleanup_interval_seconds: Cleanup every N seconds (default: 30)
        cleanup_every_n_steps: Alternative: cleanup every N training steps (overrides time-based)
        monitor_callback: Optional callback(allocated_mb, reserved_mb) for logging
        min_memory_threshold_mb: Only cleanup if reserved > threshold (default: 1000)
        verbose: Print cleanup messages (default: False)
    """

    def __init__(
        self,
        cleanup_interval_seconds: float = 30.0,
        cleanup_every_n_steps: Optional[int] = None,
        monitor_callback: Optional[Callable[[float, float], None]] = None,
        min_memory_threshold_mb: float = 1000.0,
        verbose: bool = False
    ):
        self.cleanup_interval_seconds = cleanup_interval_seconds
        self.cleanup_every_n_steps = cleanup_every_n_steps
        self.monitor_callback = monitor_callback
        self.min_memory_threshold_mb = min_memory_threshold_mb
        self.verbose = verbose

        self._stop_event = threading.Event()
        self._cleanup_event = threading.Event()  # For step-based triggers
        self._thread: Optional[threading.Thread] = None
        self._step_counter = 0
        self._lock = threading.Lock()

        # Stats
        self.cleanup_count = 0
        self.total_freed_mb = 0.0
        self.last_cleanup_time = 0.0

    def start(self) -> None:
        """Start the cleanup daemon thread."""
        if not torch.cuda.is_available():
            if self.verbose:
                print("[GPUCleanup] CUDA not available, cleanup thread disabled")
            return

        # Check for optimal allocator config
        alloc_conf = check_cuda_alloc_conf()
        if not alloc_conf['expandable_segments']:
            print("[GPUCleanup] Tip: Set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True for reduced fragmentation")

        self._stop_event.clear()
        self._cleanup_event.clear()
        self._thread = threading.Thread(
            target=self._cleanup_loop,
            daemon=True,
            name="GPUCleanupThread"
        )
        self._thread.start()

        if self.verbose:
            mode = f"every {self.cleanup_every_n_steps} steps" if self.cleanup_every_n_steps else f"every {self.cleanup_interval_seconds}s"
            print(f"[GPUCleanup] Started ({mode}, threshold={self.min_memory_threshold_mb}MB)")

    def stop(self) -> None:
        """Stop the cleanup thread gracefully."""
        self._stop_event.set()
        self._cleanup_event.set()  # Wake up thread if waiting

        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5.0)

        if self.verbose:
            print(f"[GPUCleanup] Stopped. Cleanups: {self.cleanup_count}, "
                  f"Total freed: {self.total_freed_mb:.1f}MB")

    def notify_step(self) -> None:
        """
        Notify thread that a training step completed.
        Call this at end of each training step for step-based cleanup.
        """
        if self.cleanup_every_n_steps is None:
            return

        with self._lock:
            self._step_counter += 1
            if self._step_counter >= self.cleanup_every_n_steps:
                self._step_counter = 0
                self._cleanup_event.set()

    def _cleanup_loop(self) -> None:
        """Main cleanup loop running in background thread."""
        while not self._stop_event.is_set():
            try:
                if self.cleanup_every_n_steps is not None:
                    # Step-based: wait for step trigger
                    self._cleanup_event.wait()
                    self._cleanup_event.clear()
                else:
                    # Time-based: wait for interval
                    self._stop_event.wait(timeout=self.cleanup_interval_seconds)

                if self._stop_event.is_set():
                    break

                self._do_cleanup()

            except Exception as e:
                if self.verbose:
                    print(f"[GPUCleanup] Error in cleanup loop: {e}")

    def _do_cleanup(self) -> float:
        """
        Perform actual memory cleanup.

        Returns:
            MB freed by cleanup
        """
        if not torch.cuda.is_available():
            return 0.0

        # Check memory threshold
        reserved_mb = torch.cuda.memory_reserved() / 1024 / 1024

        if reserved_mb < self.min_memory_threshold_mb:
            return 0.0

        # Measure before
        allocated_before = torch.cuda.memory_allocated() / 1024 / 1024
        reserved_before = reserved_mb

        # Sync GPU to ensure operations complete before cleanup
        torch.cuda.synchronize()

        # Python garbage collection (releases PyTorch tensors with no refs)
        gc.collect()

        # PyTorch CUDA cache cleanup (returns memory to CUDA driver)
        torch.cuda.empty_cache()

        # Measure after
        allocated_after = torch.cuda.memory_allocated() / 1024 / 1024
        reserved_after = torch.cuda.memory_reserved() / 1024 / 1024

        freed_mb = reserved_before - reserved_after
        self.total_freed_mb += max(0, freed_mb)
        self.cleanup_count += 1
        self.last_cleanup_time = time.time()

        if self.verbose and freed_mb > 0:
            print(f"[GPUCleanup] Freed {freed_mb:.1f}MB "
                  f"(reserved: {reserved_before:.0f}→{reserved_after:.0f}MB)")

        # Call monitor callback if provided
        if self.monitor_callback:
            self.monitor_callback(allocated_after, reserved_after)

        return max(0, freed_mb)

    def force_cleanup(self) -> float:
        """
        Force immediate cleanup (call from main thread).
        Bypasses threshold check.

        Returns:
            MB freed by cleanup
        """
        if not torch.cuda.is_available():
            return 0.0

        reserved_before = torch.cuda.memory_reserved() / 1024 / 1024

        torch.cuda.synchronize()
        gc.collect()
        torch.cuda.empty_cache()

        reserved_after = torch.cuda.memory_reserved() / 1024 / 1024
        freed = reserved_before - reserved_after

        self.cleanup_count += 1
        self.total_freed_mb += max(0, freed)
        self.last_cleanup_time = time.time()

        if self.verbose:
            print(f"[GPUCleanup] Force cleanup freed {freed:.1f}MB")

        return max(0, freed)

    def get_stats(self) -> Dict[str, float]:
        """Get cleanup statistics."""
        return {
            'cleanup_count': self.cleanup_count,
            'total_freed_mb': self.total_freed_mb,
            'avg_freed_mb': self.total_freed_mb / max(1, self.cleanup_count),
            'last_cleanup_time': self.last_cleanup_time,
        }

    def get_memory_info(self) -> Dict[str, float]:
        """Get current GPU memory info in MB."""
        if not torch.cuda.is_available():
            return {'allocated_mb': 0, 'reserved_mb': 0, 'free_mb': 0}

        allocated = torch.cuda.memory_allocated() / 1024 / 1024
        reserved = torch.cuda.memory_reserved() / 1024 / 1024

        # Get total GPU memory
        total = torch.cuda.get_device_properties(0).total_memory / 1024 / 1024
        free = total - reserved

        return {
            'allocated_mb': allocated,
            'reserved_mb': reserved,
            'free_mb': free,
            'total_mb': total,
        }


def cleanup_gpu_memory(verbose: bool = False) -> float:
    """
    One-shot GPU memory cleanup function.

    Convenience function for manual cleanup without thread.

    Args:
        verbose: Print cleanup info

    Returns:
        MB freed
    """
    if not torch.cuda.is_available():
        return 0.0

    reserved_before = torch.cuda.memory_reserved() / 1024 / 1024

    torch.cuda.synchronize()
    gc.collect()
    torch.cuda.empty_cache()

    reserved_after = torch.cuda.memory_reserved() / 1024 / 1024
    freed = reserved_before - reserved_after

    if verbose:
        print(f"[cleanup_gpu_memory] Freed {freed:.1f}MB "
              f"(reserved: {reserved_before:.0f}→{reserved_after:.0f}MB)")

    return max(0, freed)

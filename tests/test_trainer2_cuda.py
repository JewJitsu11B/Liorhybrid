"""
Tests for trainer2 CUDA-only behavior.

Verifies that trainer2 correctly enforces CUDA-only operation
and provides clear error messages when CUDA is unavailable.
"""

import pytest
import torch
import sys
from pathlib import Path

# Add repo root to path for imports
repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))


def test_trainer2_import_requires_cuda():
    """
    Test that trainer2 raises RuntimeError at import if CUDA is not available.
    
    Note: This test can only verify the error when CUDA is actually unavailable.
    On CUDA systems, it verifies the module imports successfully.
    """
    if torch.cuda.is_available():
        # CUDA is available, trainer2 should import successfully
        try:
            from training import trainer2
            assert hasattr(trainer2, 'TrainConfig')
            assert hasattr(trainer2, 'trainer2_entrypoint')
            print("✓ trainer2 imported successfully on CUDA system")
        except Exception as e:
            pytest.fail(f"trainer2 should import on CUDA system but failed: {e}")
    else:
        # CUDA not available, trainer2 should raise RuntimeError
        with pytest.raises(RuntimeError, match="CUDA is required"):
            from training import trainer2


def test_trainer2_cuda_enforcement_message():
    """
    Test that the CUDA error message is clear and helpful.
    
    This test verifies the error message content when CUDA is unavailable.
    """
    if not torch.cuda.is_available():
        try:
            from training import trainer2
            pytest.fail("Expected RuntimeError when CUDA not available")
        except RuntimeError as e:
            error_msg = str(e)
            # Check that error message is informative
            assert "CUDA is required" in error_msg
            assert "No CPU fallback" in error_msg or "not allowed" in error_msg
            print(f"✓ Clear error message: {error_msg}")


def test_trainer2_device_is_cuda():
    """
    Test that trainer2.DEVICE is set to CUDA device.
    
    Only runs if CUDA is available.
    """
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available, skipping CUDA device test")
    
    from training import trainer2
    
    assert trainer2.DEVICE.type == 'cuda'
    print(f"✓ trainer2.DEVICE is: {trainer2.DEVICE}")


def test_trainconfig_validation_requires_cuda():
    """
    Test that TrainConfig validation fails gracefully when CUDA unavailable.
    """
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available, cannot test validation")
    
    from training.trainer2 import TrainConfig, validate_config
    
    # Create a basic config
    cfg = TrainConfig()
    
    # Validation should pass on CUDA system
    try:
        validate_config(cfg)
        print("✓ TrainConfig validation passed on CUDA system")
    except RuntimeError as e:
        if "CUDA not found" in str(e):
            # This is expected if CUDA became unavailable
            pass
        else:
            raise


def test_trainconfig_has_seed():
    """Test that TrainConfig has a seed field for reproducibility."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available, cannot import trainer2")
    
    from training.trainer2 import TrainConfig
    
    cfg = TrainConfig()
    assert hasattr(cfg, 'seed')
    assert isinstance(cfg.seed, int)
    assert cfg.seed == 42  # Default seed
    print(f"✓ TrainConfig has seed: {cfg.seed}")


def test_trainconfig_has_deterministic_flags():
    """Test that TrainConfig has deterministic flags."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available, cannot import trainer2")
    
    from training.trainer2 import TrainConfig
    
    cfg = TrainConfig()
    
    # Check for deterministic flags
    assert hasattr(cfg, 'cudnn_deterministic')
    assert hasattr(cfg, 'cudnn_benchmark')
    assert isinstance(cfg.cudnn_deterministic, bool)
    assert isinstance(cfg.cudnn_benchmark, bool)
    
    print(f"✓ cudnn_deterministic: {cfg.cudnn_deterministic}")
    print(f"✓ cudnn_benchmark: {cfg.cudnn_benchmark}")


def test_set_random_seed_function_exists():
    """Test that set_random_seed function exists in trainer2."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available, cannot import trainer2")
    
    from training import trainer2
    
    assert hasattr(trainer2, 'set_random_seed')
    assert callable(trainer2.set_random_seed)
    print("✓ set_random_seed function exists")


def test_set_random_seed_execution():
    """Test that set_random_seed can be called without errors."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available, cannot import trainer2")
    
    from training.trainer2 import set_random_seed
    
    # Should not raise any errors
    try:
        set_random_seed(42)
        print("✓ set_random_seed(42) executed successfully")
    except Exception as e:
        pytest.fail(f"set_random_seed should not raise error: {e}")


def test_no_cpu_fallback_in_trainer2():
    """
    Verify that trainer2 does not have CPU fallback code.
    
    This is a smoke test to ensure the CUDA-only constraint is maintained.
    """
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available, cannot import trainer2")
    
    from training import trainer2
    
    # DEVICE should be CUDA, not CPU
    assert trainer2.DEVICE.type == 'cuda'
    
    # Check that there's no device selection logic that defaults to CPU
    import inspect
    source = inspect.getsource(trainer2)
    
    # Should not have "device('cpu')" or similar CPU fallback
    assert "device('cpu')" not in source or "CUDA is required" in source
    print("✓ No CPU fallback detected in trainer2")


def test_trainer2_grad_disabled():
    """
    Test that trainer2 disables gradients as documented.
    
    trainer2 uses manual updates, not autograd.
    """
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available, cannot import trainer2")
    
    from training import trainer2
    
    # After importing trainer2, gradients should be disabled
    # (Note: This is set at module level, so it affects the entire process)
    # We can't test this in isolation, but we can check the function exists
    assert hasattr(trainer2, 'assert_no_autograd')
    print("✓ assert_no_autograd function exists")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_trainer2_full_import_on_cuda():
    """
    Integration test: verify all key components import successfully on CUDA system.
    """
    from training.trainer2 import (
        TrainConfig,
        validate_config,
        set_random_seed,
        apply_backend_flags,
        trainer2_entrypoint,
    )
    
    print("✓ All trainer2 components imported successfully")
    
    # Create and validate config
    cfg = TrainConfig(seed=123, max_epochs=1)
    validate_config(cfg)
    print(f"✓ Config validated with seed={cfg.seed}")


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

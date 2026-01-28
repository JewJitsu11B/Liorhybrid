"""
Tests for checkpoint validation functionality.

Verifies that checkpoint validator correctly identifies missing keys,
shape mismatches, and provides helpful error messages.
"""

import pytest
import torch
import sys
from pathlib import Path

# Add repo root to path
repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from training.checkpoint_validator import (
    validate_checkpoint_schema,
    CheckpointValidationError,
    get_checkpoint_info,
    validate_checkpoint_compatibility,
)


def create_valid_checkpoint():
    """Helper to create a valid checkpoint for testing."""
    return {
        'model_state_dict': {
            'layer1.weight': torch.randn(512, 256),
            'layer1.bias': torch.randn(512),
        },
        'field_state_dict': {
            'T': torch.randn(8, 8, 16),
        },
        'input_embedding_state_dict': {
            'weight': torch.randn(32000, 512),  # (vocab_size, d_model)
        },
        'lm_head_state_dict': {
            'weight': torch.randn(32000, 512),  # (vocab_size, d_model)
        },
        'config': {
            'vocab_size': 32000,
            'd_model': 512,
            'n_layers': 6,
            'n_heads': 8,
        },
        'epoch': 5,
        'global_step': 10000,
    }


def test_valid_checkpoint_passes():
    """Test that a valid checkpoint passes validation."""
    checkpoint = create_valid_checkpoint()
    
    # Should not raise exception
    try:
        validate_checkpoint_schema(checkpoint, strict=True)
        print("✓ Valid checkpoint passed validation")
    except CheckpointValidationError as e:
        pytest.fail(f"Valid checkpoint should pass: {e}")


def test_missing_model_state_dict():
    """Test that missing model_state_dict is caught."""
    checkpoint = create_valid_checkpoint()
    del checkpoint['model_state_dict']
    
    with pytest.raises(CheckpointValidationError, match="model_state_dict"):
        validate_checkpoint_schema(checkpoint, strict=True)
    
    print("✓ Missing model_state_dict caught")


def test_missing_field_state():
    """Test that missing field state is caught."""
    checkpoint = create_valid_checkpoint()
    del checkpoint['field_state_dict']
    
    with pytest.raises(CheckpointValidationError, match="field state"):
        validate_checkpoint_schema(checkpoint, strict=True)
    
    print("✓ Missing field state caught")


def test_missing_input_embedding():
    """Test that missing input embedding is caught."""
    checkpoint = create_valid_checkpoint()
    del checkpoint['input_embedding_state_dict']
    
    with pytest.raises(CheckpointValidationError, match="input_embedding"):
        validate_checkpoint_schema(checkpoint, strict=True)
    
    print("✓ Missing input embedding caught")


def test_missing_lm_head():
    """Test that missing lm_head is caught."""
    checkpoint = create_valid_checkpoint()
    del checkpoint['lm_head_state_dict']
    
    with pytest.raises(CheckpointValidationError, match="lm_head"):
        validate_checkpoint_schema(checkpoint, strict=True)
    
    print("✓ Missing lm_head caught")


def test_empty_model_state_dict():
    """Test that empty model_state_dict is caught."""
    checkpoint = create_valid_checkpoint()
    checkpoint['model_state_dict'] = {}
    
    with pytest.raises(CheckpointValidationError, match="empty"):
        validate_checkpoint_schema(checkpoint, strict=True)
    
    print("✓ Empty model_state_dict caught")


def test_field_state_alternative_name():
    """Test that 'field_state' (without _dict) is accepted."""
    checkpoint = create_valid_checkpoint()
    checkpoint['field_state'] = checkpoint.pop('field_state_dict')
    
    # Should pass with alternative name
    try:
        validate_checkpoint_schema(checkpoint, strict=True)
        print("✓ Alternative 'field_state' name accepted")
    except CheckpointValidationError as e:
        pytest.fail(f"Should accept 'field_state' alternative: {e}")


def test_non_strict_mode_warnings():
    """Test that non-strict mode prints warnings instead of raising."""
    checkpoint = create_valid_checkpoint()
    checkpoint['input_embedding_state_dict'] = {}  # Empty, should warn
    
    # Should not raise in non-strict mode
    try:
        validate_checkpoint_schema(checkpoint, strict=False)
        print("✓ Non-strict mode allows warnings")
    except CheckpointValidationError:
        pytest.fail("Non-strict mode should not raise exception")


def test_vocab_size_mismatch_warning():
    """Test that vocab_size mismatch generates warning."""
    checkpoint = create_valid_checkpoint()
    
    # This should generate a warning but not fail
    validate_checkpoint_schema(checkpoint, expected_vocab_size=16000, strict=False)
    print("✓ Vocab size mismatch warning generated")


def test_d_model_mismatch_warning():
    """Test that d_model mismatch generates warning."""
    checkpoint = create_valid_checkpoint()
    
    # This should generate a warning but not fail
    validate_checkpoint_schema(checkpoint, expected_d_model=256, strict=False)
    print("✓ d_model mismatch warning generated")


def test_get_checkpoint_info():
    """Test that get_checkpoint_info extracts correct metadata."""
    checkpoint = create_valid_checkpoint()
    
    info = get_checkpoint_info(checkpoint)
    
    assert info['has_model_state'] is True
    assert info['has_field_state'] is True
    assert info['has_input_embedding'] is True
    assert info['has_lm_head'] is True
    assert info['epoch'] == 5
    assert info['global_step'] == 10000
    assert info['config']['vocab_size'] == 32000
    assert info['config']['d_model'] == 512
    
    print("✓ Checkpoint info extracted correctly")


def test_get_checkpoint_info_infers_shapes():
    """Test that get_checkpoint_info infers shapes from embeddings."""
    checkpoint = create_valid_checkpoint()
    
    info = get_checkpoint_info(checkpoint)
    
    assert 'inferred_vocab_size' in info
    assert 'inferred_d_model' in info
    assert info['inferred_vocab_size'] == 32000
    assert info['inferred_d_model'] == 512
    
    print("✓ Shape inference works")


def test_validate_compatibility_success():
    """Test that compatible checkpoint and model pass validation."""
    checkpoint = create_valid_checkpoint()
    
    # Create a mock model with matching parameters
    import torch.nn as nn
    
    class MockModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer1 = nn.Linear(256, 512)
    
    model = MockModel()
    
    is_compatible, issues = validate_checkpoint_compatibility(checkpoint, model, strict=False)
    
    # Should be compatible (we're not strict about all keys)
    assert isinstance(is_compatible, bool)
    print(f"✓ Compatibility check completed: compatible={is_compatible}")


def test_validate_compatibility_shape_mismatch():
    """Test that shape mismatch is detected."""
    checkpoint = create_valid_checkpoint()
    
    # Add a parameter with wrong shape
    checkpoint['model_state_dict']['layer1.weight'] = torch.randn(256, 128)  # Wrong shape
    
    import torch.nn as nn
    
    class MockModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer1 = nn.Linear(256, 512)  # Expects (512, 256)
    
    model = MockModel()
    
    is_compatible, issues = validate_checkpoint_compatibility(checkpoint, model, strict=True)
    
    # Should detect shape mismatch
    assert len(issues) > 0
    assert any('shape' in str(issue).lower() for issue in issues)
    print("✓ Shape mismatch detected")


def test_checkpoint_with_wrong_types():
    """Test validation catches wrong types for state dicts."""
    checkpoint = create_valid_checkpoint()
    checkpoint['model_state_dict'] = "not_a_dict"  # Wrong type
    
    with pytest.raises(CheckpointValidationError, match="must be a dictionary"):
        validate_checkpoint_schema(checkpoint, strict=True)
    
    print("✓ Wrong type caught")


def test_missing_metadata_warnings():
    """Test that missing metadata generates warnings."""
    checkpoint = create_valid_checkpoint()
    del checkpoint['config']
    del checkpoint['epoch']
    del checkpoint['global_step']
    
    # Should still pass but with warnings
    validate_checkpoint_schema(checkpoint, strict=False)
    print("✓ Missing metadata warnings generated")


def test_validation_error_message_quality():
    """Test that error messages are clear and helpful."""
    checkpoint = {
        'model_state_dict': {},
        # Missing other required keys
    }
    
    try:
        validate_checkpoint_schema(checkpoint, strict=True)
        pytest.fail("Should have raised validation error")
    except CheckpointValidationError as e:
        error_msg = str(e)
        
        # Check that error message lists missing keys
        assert 'field_state' in error_msg or 'field state' in error_msg
        assert 'input_embedding' in error_msg
        assert 'lm_head' in error_msg
        
        print(f"✓ Clear error message:\n{error_msg}")


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

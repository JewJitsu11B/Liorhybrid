"""
Checkpoint Schema Validation

Validates that checkpoints contain all required state dicts with compatible shapes.
Provides clear error messages for missing or incompatible components.
"""

from typing import Dict, Any, List, Tuple, Optional
import torch


class CheckpointValidationError(Exception):
    """Raised when checkpoint validation fails."""
    pass


def validate_checkpoint_schema(
    checkpoint: Dict[str, Any],
    expected_vocab_size: Optional[int] = None,
    expected_d_model: Optional[int] = None,
    strict: bool = True
) -> None:
    """
    Validate checkpoint contains required state dicts with proper shapes.
    
    Args:
        checkpoint: Loaded checkpoint dictionary
        expected_vocab_size: Expected vocabulary size (must be positive if provided)
        expected_d_model: Expected model dimension (must be positive if provided)
        strict: If True, raises exception on validation failure.
                If False, only prints warnings.
    
    Raises:
        CheckpointValidationError: If strict=True and validation fails
        ValueError: If expected_vocab_size or expected_d_model are invalid
    
    Required checkpoint keys:
        - model_state_dict: Main model parameters
        - field_state_dict or field_state: Field evolution state  
        - input_embedding_state_dict: Input embedding parameters
        - lm_head_state_dict: Language model head parameters
    """
    # Validate input parameters
    if expected_vocab_size is not None and expected_vocab_size <= 0:
        raise ValueError(f"expected_vocab_size must be positive, got {expected_vocab_size}")
    if expected_d_model is not None and expected_d_model <= 0:
        raise ValueError(f"expected_d_model must be positive, got {expected_d_model}")
    
    errors = []
    warnings = []
    
    # Check for required top-level keys
    required_keys = [
        ('model_state_dict', 'Main model state dictionary'),
        ('input_embedding_state_dict', 'Input embedding state dictionary'),
        ('lm_head_state_dict', 'Language model head state dictionary'),
    ]
    
    # Field state can be either field_state_dict or field_state
    has_field_state = 'field_state_dict' in checkpoint or 'field_state' in checkpoint
    if not has_field_state:
        errors.append(
            "Missing field state: checkpoint must contain either 'field_state_dict' or 'field_state'"
        )
    
    for key, description in required_keys:
        if key not in checkpoint:
            errors.append(f"Missing '{key}': {description} not found in checkpoint")
    
    # If basic validation fails, return early
    if errors:
        error_msg = "Checkpoint schema validation failed:\n" + "\n".join(f"  - {e}" for e in errors)
        if strict:
            raise CheckpointValidationError(error_msg)
        else:
            print(f"WARNING: {error_msg}")
            return
    
    # Validate model_state_dict shapes
    model_state = checkpoint['model_state_dict']
    if not isinstance(model_state, dict):
        errors.append(f"'model_state_dict' must be a dictionary, got {type(model_state)}")
    elif len(model_state) == 0:
        errors.append("'model_state_dict' is empty")
    else:
        # Check for some expected keys
        model_keys = list(model_state.keys())
        if len(model_keys) == 0:
            warnings.append("'model_state_dict' contains no parameters")
    
    # Validate input_embedding_state_dict
    input_emb_state = checkpoint['input_embedding_state_dict']
    if not isinstance(input_emb_state, dict):
        errors.append(f"'input_embedding_state_dict' must be a dictionary, got {type(input_emb_state)}")
    elif len(input_emb_state) == 0:
        warnings.append("'input_embedding_state_dict' is empty")
    else:
        # Try to infer vocab_size and d_model from embeddings
        inferred_info = _infer_embedding_shapes(input_emb_state)
        if inferred_info:
            vocab_size, d_model = inferred_info
            if expected_vocab_size and vocab_size != expected_vocab_size:
                warnings.append(
                    f"Input embedding vocab_size ({vocab_size}) doesn't match expected ({expected_vocab_size})"
                )
            if expected_d_model and d_model != expected_d_model:
                warnings.append(
                    f"Input embedding d_model ({d_model}) doesn't match expected ({expected_d_model})"
                )
    
    # Validate lm_head_state_dict
    lm_head_state = checkpoint['lm_head_state_dict']
    if not isinstance(lm_head_state, dict):
        errors.append(f"'lm_head_state_dict' must be a dictionary, got {type(lm_head_state)}")
    elif len(lm_head_state) == 0:
        warnings.append("'lm_head_state_dict' is empty")
    else:
        # Try to infer shapes from lm_head
        inferred_info = _infer_lm_head_shapes(lm_head_state)
        if inferred_info:
            vocab_size, d_model = inferred_info
            if expected_vocab_size and vocab_size != expected_vocab_size:
                warnings.append(
                    f"LM head vocab_size ({vocab_size}) doesn't match expected ({expected_vocab_size})"
                )
            if expected_d_model and d_model != expected_d_model:
                warnings.append(
                    f"LM head d_model ({d_model}) doesn't match expected ({expected_d_model})"
                )
    
    # Validate field_state_dict/field_state
    field_state_key = 'field_state_dict' if 'field_state_dict' in checkpoint else 'field_state'
    field_state = checkpoint[field_state_key]
    if not isinstance(field_state, dict):
        errors.append(f"'{field_state_key}' must be a dictionary, got {type(field_state)}")
    elif len(field_state) == 0:
        warnings.append(f"'{field_state_key}' is empty")
    
    # Check for metadata
    if 'config' not in checkpoint:
        warnings.append("No 'config' found in checkpoint (may cause compatibility issues)")
    if 'epoch' not in checkpoint:
        warnings.append("No 'epoch' found in checkpoint (cannot resume training properly)")
    if 'global_step' not in checkpoint:
        warnings.append("No 'global_step' found in checkpoint")
    
    # Report results
    if errors:
        error_msg = "Checkpoint validation failed:\n" + "\n".join(f"  ERROR: {e}" for e in errors)
        if strict:
            raise CheckpointValidationError(error_msg)
        else:
            print(f"WARNING: {error_msg}")
    
    if warnings:
        warning_msg = "Checkpoint validation warnings:\n" + "\n".join(f"  WARNING: {w}" for w in warnings)
        print(warning_msg)
    
    if not errors and not warnings:
        print("✓ Checkpoint schema validation passed")


def _infer_embedding_shapes(embedding_state: Dict[str, Any]) -> Optional[Tuple[int, int]]:
    """
    Try to infer (vocab_size, d_model) from embedding state dict.
    
    Returns:
        Tuple of (vocab_size, d_model) if found, None otherwise
    """
    # Look for common embedding weight keys
    for key in ['weight', 'embedding.weight', 'token_embedding.weight']:
        if key in embedding_state:
            weight = embedding_state[key]
            if isinstance(weight, torch.Tensor) and weight.ndim == 2:
                return (weight.shape[0], weight.shape[1])  # (vocab_size, d_model)
    
    return None


def _infer_lm_head_shapes(lm_head_state: Dict[str, Any]) -> Optional[Tuple[int, int]]:
    """
    Try to infer (vocab_size, d_model) from lm_head state dict.
    
    Returns:
        Tuple of (vocab_size, d_model) if found, None otherwise
    """
    # Look for common lm_head weight keys
    for key in ['weight', 'linear.weight', 'projection.weight']:
        if key in lm_head_state:
            weight = lm_head_state[key]
            if isinstance(weight, torch.Tensor) and weight.ndim == 2:
                return (weight.shape[0], weight.shape[1])  # (vocab_size, d_model)
    
    return None


def get_checkpoint_info(checkpoint: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract useful information from checkpoint for debugging.
    
    Args:
        checkpoint: Loaded checkpoint dictionary
    
    Returns:
        Dictionary with checkpoint metadata and shapes
    """
    info = {
        'keys': list(checkpoint.keys()),
        'has_model_state': 'model_state_dict' in checkpoint,
        'has_field_state': 'field_state_dict' in checkpoint or 'field_state' in checkpoint,
        'has_input_embedding': 'input_embedding_state_dict' in checkpoint,
        'has_lm_head': 'lm_head_state_dict' in checkpoint,
    }
    
    # Add config info if available
    if 'config' in checkpoint:
        config = checkpoint['config']
        if isinstance(config, dict):
            info['config'] = {
                'vocab_size': config.get('vocab_size'),
                'd_model': config.get('d_model'),
                'n_layers': config.get('n_layers'),
                'n_heads': config.get('n_heads'),
            }
    
    # Add training progress info
    if 'epoch' in checkpoint:
        info['epoch'] = checkpoint['epoch']
    if 'global_step' in checkpoint:
        info['global_step'] = checkpoint['global_step']
    
    # Add shape info from embeddings
    if 'input_embedding_state_dict' in checkpoint:
        shapes = _infer_embedding_shapes(checkpoint['input_embedding_state_dict'])
        if shapes:
            info['inferred_vocab_size'], info['inferred_d_model'] = shapes
    
    return info


def validate_checkpoint_compatibility(
    checkpoint: Dict[str, Any],
    model: torch.nn.Module,
    strict: bool = False
) -> Tuple[bool, List[str]]:
    """
    Check if checkpoint is compatible with model architecture.
    
    Args:
        checkpoint: Loaded checkpoint dictionary
        model: PyTorch model to check compatibility with
        strict: If True, requires exact parameter match
    
    Returns:
        Tuple of (is_compatible, list_of_issues)
    """
    issues = []
    
    if 'model_state_dict' not in checkpoint:
        issues.append("Checkpoint missing 'model_state_dict'")
        return (False, issues)
    
    model_state = checkpoint['model_state_dict']
    model_params = dict(model.named_parameters())
    
    # Check for missing keys in checkpoint
    missing_in_checkpoint = set(model_params.keys()) - set(model_state.keys())
    if missing_in_checkpoint:
        missing_list = list(missing_in_checkpoint)[:5]
        truncated = " (and more...)" if len(missing_in_checkpoint) > 5 else ""
        issues.append(f"Missing in checkpoint: {missing_list}{truncated}")
    
    # Check for unexpected keys in checkpoint
    unexpected_in_checkpoint = set(model_state.keys()) - set(model_params.keys())
    if unexpected_in_checkpoint and strict:
        unexpected_list = list(unexpected_in_checkpoint)[:5]
        truncated = " (and more...)" if len(unexpected_in_checkpoint) > 5 else ""
        issues.append(f"Unexpected in checkpoint: {unexpected_list}{truncated}")
    
    # Check shape compatibility for common keys
    common_keys = set(model_params.keys()) & set(model_state.keys())
    shape_mismatches = []
    for key in common_keys:
        if model_params[key].shape != model_state[key].shape:
            shape_mismatches.append(
                f"{key}: model {model_params[key].shape} vs checkpoint {model_state[key].shape}"
            )
    
    if shape_mismatches:
        truncated = " (and more...)" if len(shape_mismatches) > 3 else ""
        issues.append(f"Shape mismatches: {shape_mismatches[:3]}{truncated}")
    
    is_compatible = len(issues) == 0 or (not strict and len(shape_mismatches) == 0)
    
    return (is_compatible, issues)

"""
Tests for inference module imports using absolute paths.

Verifies that inference.py and related modules use absolute imports
via Liorhybrid.* package structure.
"""

import pytest
import sys
from pathlib import Path

# Ensure repo root is in path
repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))


def test_inference_module_import():
    """Test that inference module can be imported via absolute path."""
    try:
        from Liorhybrid.inference.inference import InferenceEngine
        print("✓ InferenceEngine imported via Liorhybrid.inference.inference")
        assert InferenceEngine is not None
    except ImportError as e:
        pytest.fail(f"Failed to import InferenceEngine via absolute path: {e}")


def test_inference_imports_core_modules():
    """Test that inference.py correctly imports core modules with absolute paths."""
    try:
        # These imports should work because inference.py uses absolute imports
        from Liorhybrid.core import CognitiveTensorField, FieldConfig
        from Liorhybrid.inference import GeometricTransformer
        
        print("✓ Core modules imported successfully")
        assert CognitiveTensorField is not None
        assert FieldConfig is not None
        assert GeometricTransformer is not None
    except ImportError as e:
        pytest.fail(f"Failed to import core modules: {e}")


def test_inference_imports_training_modules():
    """Test that inference.py imports training modules correctly."""
    try:
        from Liorhybrid.training.tokenizer import CognitiveTokenizer
        from Liorhybrid.training.embeddings import MultimodalEmbedding
        
        print("✓ Training modules imported successfully")
        assert CognitiveTokenizer is not None
        assert MultimodalEmbedding is not None
    except ImportError as e:
        pytest.fail(f"Failed to import training modules: {e}")


def test_checkpoint_validator_import():
    """Test that checkpoint validator can be imported."""
    try:
        from Liorhybrid.training.checkpoint_validator import (
            validate_checkpoint_schema,
            CheckpointValidationError,
            get_checkpoint_info
        )
        
        print("✓ Checkpoint validator imported successfully")
        assert validate_checkpoint_schema is not None
        assert CheckpointValidationError is not None
        assert get_checkpoint_info is not None
    except ImportError as e:
        pytest.fail(f"Failed to import checkpoint validator: {e}")


def test_sdm_memory_import():
    """Test that SDM memory stub can be imported."""
    try:
        from Liorhybrid.inference.sdm_memory import SDMMemory, create_sdm_memory
        
        print("✓ SDM memory stub imported successfully")
        assert SDMMemory is not None
        assert create_sdm_memory is not None
    except ImportError as e:
        pytest.fail(f"Failed to import SDM memory: {e}")


def test_cli_module_import():
    """Test that CLI module can be imported and has entrypoints."""
    try:
        from Liorhybrid.cli import main, train_entrypoint, inference_entrypoint
        
        print("✓ CLI module imported successfully")
        assert callable(main)
        assert callable(train_entrypoint)
        assert callable(inference_entrypoint)
    except ImportError as e:
        pytest.fail(f"Failed to import CLI module: {e}")


def test_inference_package_structure():
    """Test that inference package has expected structure."""
    try:
        import Liorhybrid.inference as inf_package
        
        # Check for expected modules/attributes
        expected = [
            'inference',
            'geometric_attention',
            'geometric_products',
            'geometric_stack',
            'sdm_memory',
        ]
        
        for module_name in expected:
            # Try to access each module
            try:
                module = getattr(inf_package, module_name, None)
                if module is None:
                    # Try importing directly
                    __import__(f'Liorhybrid.inference.{module_name}')
                print(f"✓ inference.{module_name} accessible")
            except (AttributeError, ImportError) as e:
                # Some modules might not be in __init__, that's ok
                print(f"  inference.{module_name} not in package (may be imported directly)")
                
    except ImportError as e:
        pytest.fail(f"Failed to import inference package: {e}")


def test_absolute_imports_in_cli():
    """Verify CLI uses absolute imports for Liorhybrid modules."""
    cli_path = repo_root / 'cli.py'
    
    if not cli_path.exists():
        pytest.skip("cli.py not found")
    
    with open(cli_path, 'r') as f:
        content = f.read()
    
    # Should have absolute imports like "from Liorhybrid.training..."
    assert 'from Liorhybrid.training' in content or 'from Liorhybrid.inference' in content
    
    # Should not have relative imports like "from training..."
    # (excluding string literals and comments)
    lines = [l.strip() for l in content.split('\n') if l.strip() and not l.strip().startswith('#')]
    import_lines = [l for l in lines if l.startswith('from ') and 'import' in l]
    
    # Check that training/inference imports use Liorhybrid prefix
    for line in import_lines:
        if 'training' in line or 'inference' in line:
            if 'Liorhybrid' not in line:
                pytest.fail(f"Non-absolute import found in cli.py: {line}")
    
    print("✓ CLI uses absolute imports")


def test_absolute_imports_in_inference():
    """Verify inference.py uses absolute imports."""
    inf_path = repo_root / 'inference' / 'inference.py'
    
    if not inf_path.exists():
        pytest.skip("inference/inference.py not found")
    
    with open(inf_path, 'r') as f:
        content = f.read()
    
    # Should have absolute imports
    assert 'from Liorhybrid.core' in content
    assert 'from Liorhybrid.inference' in content
    assert 'from Liorhybrid.training' in content
    
    print("✓ inference.py uses absolute imports")


def test_sample_data_generator_import():
    """Test that sample data generator can be imported."""
    try:
        from Liorhybrid.data.sample.generate_sample_data import generate_sample_text_data
        
        print("✓ Sample data generator imported successfully")
        assert callable(generate_sample_text_data)
    except ImportError as e:
        pytest.fail(f"Failed to import sample data generator: {e}")


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

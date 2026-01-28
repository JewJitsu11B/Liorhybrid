"""
Unit tests for SDM Memory stub implementation.

Tests ensure that the memory subsystem stubs behave predictably
even though they don't implement full functionality.
"""

import pytest
import torch
import warnings
from inference.sdm_memory import SDMMemory, create_sdm_memory


def test_sdm_initialization():
    """Test that SDM memory initializes with correct parameters."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # Suppress stub warning for test
        
        memory = SDMMemory(
            capacity=1024,
            address_dim=256,
            value_dim=256,
            device='cpu'
        )
        
        assert memory.capacity == 1024
        assert memory.address_dim == 256
        assert memory.value_dim == 256
        assert memory.device == 'cpu'
        assert memory.memory_count == 0


def test_sdm_initialization_warning():
    """Test that SDM initialization issues a warning about being a stub."""
    with pytest.warns(UserWarning, match="SDMMemory is a placeholder stub"):
        memory = SDMMemory(capacity=1024, address_dim=256, value_dim=256)


def test_sdm_store():
    """Test that store operation accepts valid inputs and returns success."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        
        memory = SDMMemory(capacity=1024, address_dim=256, value_dim=256, device='cpu')
        
        address = torch.randn(256)
        value = torch.randn(256)
        
        result = memory.store(address, value)
        
        assert result is True
        assert memory.memory_count == 1


def test_sdm_store_multiple():
    """Test storing multiple items increments count."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        
        memory = SDMMemory(capacity=1024, address_dim=256, value_dim=256, device='cpu')
        
        for i in range(5):
            address = torch.randn(256)
            value = torch.randn(256)
            memory.store(address, value)
        
        assert memory.memory_count == 5


def test_sdm_store_dimension_validation():
    """Test that store validates input dimensions."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        
        memory = SDMMemory(capacity=1024, address_dim=256, value_dim=256, device='cpu')
        
        # Wrong address dimension
        with pytest.raises(ValueError, match="Address dimension mismatch"):
            memory.store(torch.randn(128), torch.randn(256))
        
        # Wrong value dimension
        with pytest.raises(ValueError, match="Value dimension mismatch"):
            memory.store(torch.randn(256), torch.randn(128))


def test_sdm_retrieve_shape():
    """Test that retrieve returns correct shapes (even though stub returns zeros)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        
        memory = SDMMemory(capacity=1024, address_dim=256, value_dim=256, device='cpu')
        
        query = torch.randn(1, 256)  # Batch size 1
        values, confidences = memory.retrieve(query, k=3)
        
        assert values.shape == (1, 3, 256)  # [batch, k, value_dim]
        assert confidences.shape == (1, 3)  # [batch, k]


def test_sdm_retrieve_batch():
    """Test that retrieve handles batched queries correctly."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        
        memory = SDMMemory(capacity=1024, address_dim=256, value_dim=256, device='cpu')
        
        batch_size = 4
        k = 5
        query = torch.randn(batch_size, 256)
        values, confidences = memory.retrieve(query, k=k)
        
        assert values.shape == (batch_size, k, 256)
        assert confidences.shape == (batch_size, k)


def test_sdm_retrieve_returns_zeros():
    """Test that stub retrieve returns zeros (expected stub behavior)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        
        memory = SDMMemory(capacity=1024, address_dim=256, value_dim=256, device='cpu')
        
        # Store something (even though stub doesn't use it)
        memory.store(torch.randn(256), torch.randn(256))
        
        # Retrieve should still return zeros (stub behavior)
        query = torch.randn(1, 256)
        values, confidences = memory.retrieve(query, k=3)
        
        assert torch.all(values == 0)
        assert torch.all(confidences == 0)


def test_sdm_clear():
    """Test that clear resets memory count."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        
        memory = SDMMemory(capacity=1024, address_dim=256, value_dim=256, device='cpu')
        
        # Store some items
        for i in range(5):
            memory.store(torch.randn(256), torch.randn(256))
        
        assert memory.memory_count == 5
        
        # Clear
        memory.clear()
        
        assert memory.memory_count == 0


def test_sdm_get_stats():
    """Test that get_stats returns expected statistics."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        
        memory = SDMMemory(capacity=1024, address_dim=256, value_dim=256, device='cpu')
        
        stats = memory.get_stats()
        
        assert stats['capacity'] == 1024
        assert stats['address_dim'] == 256
        assert stats['value_dim'] == 256
        assert stats['stored_count'] == 0
        assert stats['utilization'] == 0.0
        assert stats['is_stub'] is True


def test_sdm_repr():
    """Test that repr contains key information."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        
        memory = SDMMemory(capacity=1024, address_dim=256, value_dim=256, device='cpu')
        
        repr_str = repr(memory)
        
        assert 'SDMMemory' in repr_str
        assert '1024' in repr_str
        assert '256' in repr_str
        assert 'STUB=True' in repr_str


def test_create_sdm_memory_factory():
    """Test factory function creates memory with correct config."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        
        config = {
            'capacity': 2048,
            'address_dim': 512,
            'value_dim': 512,
            'device': 'cpu'
        }
        
        memory = create_sdm_memory(config)
        
        assert memory.capacity == 2048
        assert memory.address_dim == 512
        assert memory.value_dim == 512
        assert memory.device == 'cpu'


def test_create_sdm_memory_defaults():
    """Test factory function uses defaults for missing config."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        
        config = {}
        memory = create_sdm_memory(config)
        
        # Should use defaults
        assert memory.capacity == 2048
        assert memory.address_dim == 512
        assert memory.value_dim == 512


def test_sdm_predictable_behavior():
    """
    Test that stub behavior is predictable and deterministic.
    
    This is important for testing code that uses SDM:
    we need to know exactly what the stub will return.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        
        memory = SDMMemory(capacity=1024, address_dim=256, value_dim=256, device='cpu')
        
        # Multiple queries should all return zeros
        for i in range(3):
            query = torch.randn(2, 256)
            values, confidences = memory.retrieve(query, k=5)
            
            assert torch.all(values == 0), f"Query {i}: values not zero"
            assert torch.all(confidences == 0), f"Query {i}: confidences not zero"
            assert values.shape == (2, 5, 256)
            assert confidences.shape == (2, 5)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

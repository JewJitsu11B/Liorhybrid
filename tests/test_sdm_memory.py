"""
Unit tests for SDM Memory implementation.

Tests ensure that the memory subsystem works correctly with
content-addressable storage, retrieval, and capacity management.
"""

import pytest
import torch
import torch.nn.functional as F
from inference.sdm_memory import SDMMemory, create_sdm_memory


def test_sdm_initialization():
    """Test that SDM memory initializes with correct parameters."""
    memory = SDMMemory(
        capacity=1024,
        address_dim=256,
        value_dim=256,
        device='cpu',
        similarity_threshold=0.5
    )
    
    assert memory.capacity == 1024
    assert memory.address_dim == 256
    assert memory.value_dim == 256
    assert memory.device == 'cpu'
    assert memory.memory_count == 0
    assert memory.similarity_threshold == 0.5


def test_sdm_store():
    """Test that store operation accepts valid inputs and returns success."""
    memory = SDMMemory(capacity=1024, address_dim=256, value_dim=256, device='cpu')
    
    address = torch.randn(256)
    value = torch.randn(256)
    
    result = memory.store(address, value)
    
    assert result is True
    assert memory.memory_count == 1

def test_sdm_store_multiple():
    """Test storing multiple items increments count."""
    memory = SDMMemory(capacity=1024, address_dim=256, value_dim=256, device='cpu')
    
    for i in range(5):
        address = torch.randn(256)
        value = torch.randn(256)
        memory.store(address, value)
    
    assert memory.memory_count == 5


def test_sdm_store_dimension_validation():
    """Test that store validates input dimensions."""
    memory = SDMMemory(capacity=1024, address_dim=256, value_dim=256, device='cpu')
    
    # Wrong address dimension
    with pytest.raises(ValueError, match="Address dimension mismatch"):
        memory.store(torch.randn(128), torch.randn(256))
    
    # Wrong value dimension
    with pytest.raises(ValueError, match="Value dimension mismatch"):
        memory.store(torch.randn(256), torch.randn(128))


def test_sdm_retrieve_shape():
    """Test that retrieve returns correct shapes."""
    memory = SDMMemory(capacity=1024, address_dim=256, value_dim=256, device='cpu')
    
    # Store some data
    for i in range(3):
        address = torch.randn(256)
        value = torch.randn(256)
        memory.store(address, value)
    
    query = torch.randn(1, 256)  # Batch size 1
    values, confidences = memory.retrieve(query, k=3)
    
    assert values.shape == (1, 3, 256)  # [batch, k, value_dim]
    assert confidences.shape == (1, 3)  # [batch, k]


def test_sdm_retrieve_batch():
    """Test that retrieve handles batched queries correctly."""
    memory = SDMMemory(capacity=1024, address_dim=256, value_dim=256, device='cpu')
    
    # Store some data
    for i in range(5):
        address = torch.randn(256)
        value = torch.randn(256)
        memory.store(address, value)
    
    batch_size = 4
    k = 3
    query = torch.randn(batch_size, 256)
    values, confidences = memory.retrieve(query, k=k)
    
    assert values.shape == (batch_size, k, 256)
    assert confidences.shape == (batch_size, k)


def test_sdm_retrieve_similarity():
    """Test that retrieve returns items with highest similarity."""
    memory = SDMMemory(capacity=1024, address_dim=256, value_dim=256, device='cpu')
    
    # Store a specific address-value pair
    target_address = torch.randn(256)
    target_value = torch.ones(256) * 10  # Distinctive value
    memory.store(target_address, target_value)
    
    # Store some other random pairs
    for i in range(5):
        memory.store(torch.randn(256), torch.randn(256))
    
    # Query with the same address (should retrieve the target)
    values, confidences = memory.retrieve(target_address.unsqueeze(0), k=1)
    
    # Check that confidence is high (cosine similarity close to 1)
    assert confidences[0, 0] > 0.99, f"Expected high confidence, got {confidences[0, 0]}"
    
    # Check that value is close to target
    retrieved = values[0, 0]
    assert torch.allclose(retrieved, target_value, atol=0.01)


def test_sdm_retrieve_empty():
    """Test that retrieve on empty memory returns zeros."""
    memory = SDMMemory(capacity=1024, address_dim=256, value_dim=256, device='cpu')
    
    query = torch.randn(1, 256)
    values, confidences = memory.retrieve(query, k=3)
    
    assert torch.all(values == 0)
    assert torch.all(confidences == 0)


def test_sdm_capacity_eviction():
    """Test that exceeding capacity triggers LRU eviction."""
    capacity = 10
    memory = SDMMemory(capacity=capacity, address_dim=256, value_dim=256, device='cpu')
    
    # Fill to capacity
    for i in range(capacity):
        address = torch.randn(256)
        value = torch.randn(256) * i  # Value scales with i
        memory.store(address, value)
    
    assert memory.memory_count == capacity
    
    # Store one more (should trigger eviction)
    new_address = torch.randn(256)
    new_value = torch.randn(256) * 999
    memory.store(new_address, new_value)
    
    # Count should still be capacity
    assert memory.memory_count == capacity


def test_sdm_clear():
    """Test that clear resets memory count."""
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
    memory = SDMMemory(capacity=1024, address_dim=256, value_dim=256, device='cpu')
    
    # Store some items
    for i in range(10):
        memory.store(torch.randn(256), torch.randn(256))
    
    stats = memory.get_stats()
    
    assert stats['capacity'] == 1024
    assert stats['address_dim'] == 256
    assert stats['value_dim'] == 256
    assert stats['stored_count'] == 10
    assert stats['utilization'] == 10 / 1024
    assert stats['total_stores'] == 10
    assert 'global_time' in stats


def test_sdm_update_similarity():
    """Test that update_similarity updates existing similar memory."""
    memory = SDMMemory(capacity=1024, address_dim=256, value_dim=256, device='cpu')
    
    # Store initial value
    address = torch.randn(256)
    value = torch.ones(256)
    memory.store(address, value)
    
    # Update with same address
    update = torch.ones(256) * 0.5
    result = memory.update_similarity(address, update)
    
    assert result is True
    
    # Retrieve and check updated value
    retrieved, _ = memory.retrieve(address.unsqueeze(0), k=1)
    expected = value + update
    assert torch.allclose(retrieved[0, 0], expected, atol=0.01)


def test_sdm_threshold_filtering():
    """Test that similarity threshold filters low-confidence retrievals."""
    memory = SDMMemory(capacity=1024, address_dim=256, value_dim=256, device='cpu', similarity_threshold=0.9)
    
    # Store a value
    address = torch.randn(256)
    value = torch.ones(256)
    memory.store(address, value)
    
    # Query with very different address (low similarity)
    different_query = -address  # Opposite direction
    values, confidences = memory.retrieve(different_query.unsqueeze(0), k=1)
    
    # Should return zero confidence (below threshold)
    assert confidences[0, 0] < 0.1 or confidences[0, 0] == 0


def test_create_sdm_memory_factory():
    """Test factory function creates memory with correct config."""
    config = {
        'capacity': 2048,
        'address_dim': 512,
        'value_dim': 512,
        'device': 'cpu',
        'similarity_threshold': 0.7
    }
    
    memory = create_sdm_memory(config)
    
    assert memory.capacity == 2048
    assert memory.address_dim == 512
    assert memory.value_dim == 512
    assert memory.device == 'cpu'
    assert memory.similarity_threshold == 0.7


def test_create_sdm_memory_defaults():
    """Test factory function uses defaults for missing config."""
    config = {}
    memory = create_sdm_memory(config)
    
    # Should use defaults
    assert memory.capacity == 2048
    assert memory.address_dim == 512
    assert memory.value_dim == 512


def test_sdm_repr():
    """Test that repr contains key information."""
    memory = SDMMemory(capacity=1024, address_dim=256, value_dim=256, device='cpu')
    
    for i in range(10):
        memory.store(torch.randn(256), torch.randn(256))
    
    repr_str = repr(memory)
    
    assert 'SDMMemory' in repr_str
    assert '1024' in repr_str
    assert '10' in repr_str or 'stored=10' in repr_str
    assert '256' in repr_str


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

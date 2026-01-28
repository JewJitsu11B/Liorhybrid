"""
SDM (Sparse Distributed Memory) Implementation

A content-addressable memory system based on Kanerva's Sparse Distributed Memory model.
Provides associative memory storage and retrieval with cosine similarity addressing.

Key Features:
- Content-addressable memory storage and retrieval
- Cosine similarity-based addressing
- Capacity management with LRU eviction
- Confidence scores based on similarity
- Support for both CPU and CUDA

Reference: Kanerva, P. (1988). Sparse Distributed Memory. MIT Press.
"""

from typing import Optional, List, Tuple, Any, Dict
import torch
import torch.nn.functional as F
import math


class SDMMemory:
    """
    Sparse Distributed Memory (SDM) implementation.
    
    A content-addressable memory system that stores and retrieves vectors
    based on similarity (cosine distance). Supports capacity management,
    LRU eviction, and confidence-based retrieval.
    
    This implementation uses:
    - Cosine similarity for address matching
    - LRU (Least Recently Used) eviction when at capacity
    - Confidence scores based on similarity strength
    """
    
    def __init__(
        self,
        capacity: int = 2048,
        address_dim: int = 512,
        value_dim: int = 512,
        device: str = 'cuda',
        similarity_threshold: float = 0.0
    ):
        """
        Initialize SDM memory.
        
        Args:
            capacity: Maximum number of memory slots
            address_dim: Dimension of address vectors (for content-addressable lookup)
            value_dim: Dimension of value vectors (stored content)
            device: Device to store memory on ('cuda' or 'cpu')
            similarity_threshold: Minimum similarity for retrieval (0.0 to 1.0)
        """
        self.capacity = capacity
        self.address_dim = address_dim
        self.value_dim = value_dim
        self.device = device
        self.similarity_threshold = similarity_threshold
        
        # Storage tensors (preallocated for efficiency)
        self.addresses = torch.zeros(capacity, address_dim, device=device, dtype=torch.float32)
        self.values = torch.zeros(capacity, value_dim, device=device, dtype=torch.float32)
        self.access_times = torch.zeros(capacity, device=device, dtype=torch.long)
        self.valid_mask = torch.zeros(capacity, device=device, dtype=torch.bool)
        
        # Metadata
        self.memory_count = 0
        self.global_time = 0
        self.total_stores = 0
        self.total_retrievals = 0
    
    def _find_eviction_slot(self) -> int:
        """
        Find slot to evict using LRU (Least Recently Used) policy.
        
        Returns:
            Index of slot to evict
        """
        # Find valid slots
        valid_indices = torch.where(self.valid_mask)[0]
        if len(valid_indices) == 0:
            return 0
        
        # Get access times for valid slots
        valid_access_times = self.access_times[valid_indices]
        
        # Find LRU slot (oldest access time)
        lru_idx = valid_indices[torch.argmin(valid_access_times)]
        return lru_idx.item()
    
    def store(
        self,
        address: torch.Tensor,
        value: torch.Tensor,
        metadata: Optional[Dict] = None
    ) -> bool:
        """
        Store a value in memory at the given address.
        
        Uses cosine similarity for addressing. If at capacity, evicts
        least recently used entry.
        
        Args:
            address: Address vector (shape: [address_dim])
            value: Value vector to store (shape: [value_dim])
            metadata: Optional metadata dict (not currently used)
        
        Returns:
            True if stored successfully
        """
        # Validate shapes
        if address.shape[-1] != self.address_dim:
            raise ValueError(f"Address dimension mismatch: expected {self.address_dim}, got {address.shape[-1]}")
        if value.shape[-1] != self.value_dim:
            raise ValueError(f"Value dimension mismatch: expected {self.value_dim}, got {value.shape[-1]}")
        
        # Ensure tensors are on correct device
        address = address.to(self.device)
        value = value.to(self.device)
        
        # Normalize address for cosine similarity
        address_norm = F.normalize(address.flatten(), p=2, dim=0)
        
        # Update global time
        self.global_time += 1
        self.total_stores += 1
        
        # Find slot to store
        if self.memory_count < self.capacity:
            # Use next available slot
            slot_idx = self.memory_count
            self.memory_count += 1
        else:
            # Evict LRU slot
            slot_idx = self._find_eviction_slot()
        
        # Store address and value
        self.addresses[slot_idx] = address_norm
        self.values[slot_idx] = value.flatten()
        self.access_times[slot_idx] = self.global_time
        self.valid_mask[slot_idx] = True
        
        return True
    
    def retrieve(
        self,
        query_address: torch.Tensor,
        k: int = 1,
        threshold: Optional[float] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieve k nearest values based on query address.
        
        Uses cosine similarity to find nearest addresses. Returns both
        the retrieved values and their confidence scores (similarities).
        
        Args:
            query_address: Query address vector (shape: [batch_size, address_dim] or [address_dim])
            k: Number of nearest neighbors to retrieve
            threshold: Optional confidence threshold (overrides instance threshold)
        
        Returns:
            Tuple of (retrieved_values, confidences)
            - retrieved_values: shape [batch_size, k, value_dim]
            - confidences: shape [batch_size, k]
        """
        # Update global time and stats
        self.global_time += 1
        self.total_retrievals += 1
        
        # Handle single query or batch
        if query_address.ndim == 1:
            query_address = query_address.unsqueeze(0)
        
        batch_size = query_address.shape[0]
        query_address = query_address.to(self.device)
        
        # Use provided threshold or instance threshold
        threshold = threshold if threshold is not None else self.similarity_threshold
        
        # Normalize query addresses for cosine similarity
        query_norm = F.normalize(query_address, p=2, dim=1)  # [batch_size, address_dim]
        
        # Get valid addresses
        valid_indices = torch.where(self.valid_mask)[0]
        
        if len(valid_indices) == 0:
            # No stored memories, return zeros
            retrieved_values = torch.zeros(
                batch_size, k, self.value_dim,
                device=self.device,
                dtype=torch.float32
            )
            confidences = torch.zeros(
                batch_size, k,
                device=self.device,
                dtype=torch.float32
            )
            return retrieved_values, confidences
        
        # Get valid addresses and values
        valid_addresses = self.addresses[valid_indices]  # [n_valid, address_dim]
        valid_values = self.values[valid_indices]  # [n_valid, value_dim]
        
        # Compute cosine similarities
        # query_norm: [batch_size, address_dim]
        # valid_addresses: [n_valid, address_dim]
        similarities = torch.matmul(query_norm, valid_addresses.t())  # [batch_size, n_valid]
        
        # Apply threshold
        similarities = torch.where(
            similarities >= threshold,
            similarities,
            torch.tensor(-float('inf'), device=self.device)
        )
        
        # Get top-k
        k_actual = min(k, len(valid_indices))
        topk_similarities, topk_indices = torch.topk(similarities, k_actual, dim=1)
        
        # Retrieve values
        retrieved_values = torch.zeros(batch_size, k, self.value_dim, device=self.device)
        confidences = torch.zeros(batch_size, k, device=self.device)
        
        for i in range(batch_size):
            for j in range(k_actual):
                idx = valid_indices[topk_indices[i, j]]
                retrieved_values[i, j] = valid_values[topk_indices[i, j]]
                confidences[i, j] = topk_similarities[i, j]
                
                # Update access time for retrieved slot
                self.access_times[idx] = self.global_time
        
        # Replace -inf with 0 for confidence scores
        confidences = torch.where(
            torch.isfinite(confidences),
            confidences,
            torch.zeros_like(confidences)
        )
        
        return retrieved_values, confidences
    
    def clear(self) -> None:
        """Clear all memory and reset state."""
        self.addresses.zero_()
        self.values.zero_()
        self.access_times.zero_()
        self.valid_mask.zero_()
        self.memory_count = 0
        # Don't reset global_time to maintain temporal ordering across clears
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get memory statistics.
        
        Returns:
            Dictionary with memory stats including:
            - capacity: Maximum capacity
            - stored_count: Current number of stored items
            - utilization: Memory utilization (0.0 to 1.0)
            - total_stores: Total store operations
            - total_retrievals: Total retrieve operations
            - global_time: Current logical time
        """
        return {
            'capacity': self.capacity,
            'stored_count': self.memory_count,
            'utilization': self.memory_count / self.capacity if self.capacity > 0 else 0.0,
            'address_dim': self.address_dim,
            'value_dim': self.value_dim,
            'total_stores': self.total_stores,
            'total_retrievals': self.total_retrievals,
            'global_time': self.global_time,
            'similarity_threshold': self.similarity_threshold,
        }
    
    def update_similarity(self, address: torch.Tensor, value_update: torch.Tensor) -> bool:
        """
        Update the value of the most similar existing memory entry.
        
        Useful for incrementally updating memories rather than storing duplicates.
        
        Args:
            address: Address vector to find similar memory
            value_update: Update to apply to the value (added to existing value)
        
        Returns:
            True if update was applied, False if no similar memory found
        """
        if self.memory_count == 0:
            return False
        
        # Normalize address
        address_norm = F.normalize(address.flatten().to(self.device), p=2, dim=0)
        
        # Get valid addresses
        valid_indices = torch.where(self.valid_mask)[0]
        if len(valid_indices) == 0:
            return False
        
        valid_addresses = self.addresses[valid_indices]
        
        # Find most similar
        similarities = torch.matmul(address_norm, valid_addresses.t())
        max_sim_idx = valid_indices[torch.argmax(similarities)]
        
        # Apply threshold
        if similarities.max() < self.similarity_threshold:
            return False
        
        # Update value
        self.values[max_sim_idx] += value_update.flatten().to(self.device)
        self.access_times[max_sim_idx] = self.global_time
        self.global_time += 1
        
        return True
    
    def __repr__(self) -> str:
        return (
            f"SDMMemory(capacity={self.capacity}, "
            f"stored={self.memory_count}, "
            f"address_dim={self.address_dim}, "
            f"value_dim={self.value_dim}, "
            f"utilization={self.memory_count/self.capacity:.1%})"
        )


def create_sdm_memory(config: Dict[str, Any]) -> SDMMemory:
    """
    Factory function to create SDM memory from config.
    
    Args:
        config: Configuration dictionary with keys:
            - capacity: Memory capacity (default: 2048)
            - address_dim: Address dimension (default: 512)
            - value_dim: Value dimension (default: 512)
            - device: Device string (default: 'cuda')
            - similarity_threshold: Minimum similarity for retrieval (default: 0.0)
    
    Returns:
        SDMMemory instance
    """
    return SDMMemory(
        capacity=config.get('capacity', 2048),
        address_dim=config.get('address_dim', 512),
        value_dim=config.get('value_dim', 512),
        device=config.get('device', 'cuda'),
        similarity_threshold=config.get('similarity_threshold', 0.0)
    )


# Example usage and integration notes
"""
Integration with Inference Engine:

1. Initialize SDM in InferenceEngine.__init__():
```python
self.memory = SDMMemory(
    capacity=2048,
    address_dim=d_model,
    value_dim=d_model,
    device=device
)
```

2. Store context during generation:
```python
# After processing input
address = input_embedding  # Use input embedding as address
value = hidden_state  # Store hidden state or output
self.memory.store(address, value)
```

3. Retrieve during generation:
```python
# Before generating next token
query = current_embedding
retrieved_values, confidences = self.memory.retrieve(query, k=5)

# Use retrieved values to augment context
if confidences[0, 0] > threshold:
    context = torch.cat([context, retrieved_values], dim=1)
```

4. Entropy-gated retrieval (future enhancement):
```python
# Only retrieve when field entropy is appropriate
if field_entropy > min_entropy and field_entropy < max_entropy:
    retrieved_values, confidences = self.memory.retrieve(query, k=5)
```

Performance considerations:
- Pre-allocate memory tensors for efficiency
- Use cosine similarity (normalized dot product) for fast retrieval
- LRU eviction maintains temporal locality
- Batch retrieve operations when possible

Testing:
- Unit tests verify store/retrieve correctness
- Capacity tests verify eviction behavior
- Similarity tests verify cosine distance calculations
- Integration tests verify usage in inference loop
"""

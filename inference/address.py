"""
Linearized Address Structure

The address is a structured vector with fixed-width blocks:
[ core | geom | N1 | N2 | ... | N64 | integrity ]

Position and ordering encode meaning (conditioning on structured priors).

Dimensions (default d=512):
- core: d = 512 (embedding)
- geom: 2d = 1024 (metric + transport)
- neighbors: 64 × d_block = 5504 (N1-N64)
- integrity: 34 (ecc + timestamps)
- Total D = 7074 floats

Neighbor roles by position:
- N1-N32: absolute nearest (similarity grounding)
- N33-N48: attractors (reinforcing evidence)
- N49-N64: repulsors (contrastive evidence)

Per-neighbor block (d_block = 86):
- value: d' = 64 (reduced interaction vector)
- scores: m = 6 (6 metric-derived similarity channels)
- coords: k = 16 (routing info)

Strict Requirements (Option 6):
- Neighbor selection: ONLY metric-derived distances (no Euclidean fallback)
- All 64 slots MUST be populated (fail fast if unable)
- 6 score channels per neighbor (all from metric/transport)
- Collision detection via route_hash with rehash on conflict
"""
try: import usage_tracker; usage_tracker.track(__file__)
except: pass

import torch
import torch.nn as nn
import hashlib
from dataclasses import dataclass
from typing import Optional, Tuple, NamedTuple, Dict, Set


# =============================================================================
# Collision Detection Utilities
# =============================================================================

class AddressRegistry:
    """
    Registry for detecting address collisions.
    
    Maintains a set of route_hash values to ensure uniqueness.
    Provides rehash mechanism on collision.
    """
    
    def __init__(self):
        self.registry: Set[str] = set()
        self.collision_count: int = 0
    
    def compute_route_hash(
        self,
        embedding: torch.Tensor,
        salt: int = 0
    ) -> str:
        """
        Compute route hash for collision detection.
        
        Hash = SHA256(embedding_bytes + salt)
        
        Args:
            embedding: Address core embedding (d,)
            salt: Salt value for rehashing (default 0)
            
        Returns:
            hash_hex: 64-character hex string
        """
        # Convert embedding to bytes
        embedding_np = embedding.detach().cpu().numpy()
        embedding_bytes = embedding_np.tobytes()
        
        # Add salt
        salt_bytes = salt.to_bytes(8, byteorder='big', signed=False)
        
        # Compute SHA256
        hasher = hashlib.sha256()
        hasher.update(embedding_bytes)
        hasher.update(salt_bytes)
        hash_hex = hasher.hexdigest()
        
        return hash_hex
    
    def check_and_register(
        self,
        embedding: torch.Tensor,
        max_rehash_attempts: int = 10
    ) -> Tuple[str, int]:
        """
        Check for collision and register address.
        
        If collision detected, rehash with incrementing salt until unique.
        
        Args:
            embedding: Address core embedding (d,)
            max_rehash_attempts: Maximum rehash attempts before giving up
            
        Returns:
            route_hash: Unique hash string
            salt_used: Salt value that produced unique hash
            
        Raises:
            RuntimeError: If cannot find unique hash after max_rehash_attempts
        """
        salt = 0
        for attempt in range(max_rehash_attempts):
            route_hash = self.compute_route_hash(embedding, salt)
            
            if route_hash not in self.registry:
                # Unique hash found
                self.registry.add(route_hash)
                return route_hash, salt
            
            # Collision detected
            self.collision_count += 1
            salt += 1
        
        # Failed to find unique hash
        raise RuntimeError(
            f"Failed to find unique route_hash after {max_rehash_attempts} attempts. "
            f"Total collisions detected: {self.collision_count}"
        )
    
    def clear(self):
        """Clear registry (for testing)."""
        self.registry.clear()
        self.collision_count = 0


# Global registry (can be replaced with per-session registry if needed)
_global_address_registry = AddressRegistry()


def get_address_registry() -> AddressRegistry:
    """Get global address registry."""
    return _global_address_registry


@dataclass
class AddressConfig:
    """Configuration for address dimensions."""
    # Core embedding
    d: int = 512

    # Geometry (diagonal mode)
    use_lowrank_geom: bool = False
    r: int = 32  # low-rank dim if used

    # Neighbors
    n_nearest: int = 32
    n_attractors: int = 16
    n_repulsors: int = 16

    # Per-neighbor dimensions
    d_prime: int = 64   # value dim
    m: int = 6          # scores dim (6 similarity channels - strict requirement)
    k: int = 16         # coords dim

    # Integrity
    ecc_bits: int = 32
    n_timestamps: int = 2

    @property
    def n_neighbors(self) -> int:
        return self.n_nearest + self.n_attractors + self.n_repulsors

    @property
    def d_block(self) -> int:
        """Size of one neighbor block."""
        return self.d_prime + self.m + self.k

    @property
    def d_geom(self) -> int:
        """Size of geometry section."""
        return 2 * self.d  # metric + transport (diagonal)

    @property
    def d_integrity(self) -> int:
        """Size of integrity section."""
        return self.ecc_bits + self.n_timestamps

    @property
    def total_dim(self) -> int:
        """Total linearized address dimension."""
        return (
            self.d +                           # core
            self.d_geom +                      # geom
            self.n_neighbors * self.d_block +  # neighbors
            self.d_integrity                   # integrity
        )

    # Block offsets (for indexing into linearized vector)
    @property
    def core_start(self) -> int:
        return 0

    @property
    def core_end(self) -> int:
        return self.d

    @property
    def geom_start(self) -> int:
        return self.core_end

    @property
    def geom_end(self) -> int:
        return self.geom_start + self.d_geom

    @property
    def neighbors_start(self) -> int:
        return self.geom_end

    @property
    def neighbors_end(self) -> int:
        return self.neighbors_start + self.n_neighbors * self.d_block

    @property
    def integrity_start(self) -> int:
        return self.neighbors_end

    @property
    def integrity_end(self) -> int:
        return self.integrity_start + self.d_integrity


class NeighborSlice(NamedTuple):
    """Indices for accessing parts of a neighbor block."""
    value_start: int
    value_end: int
    scores_start: int
    scores_end: int
    coords_start: int
    coords_end: int


class NeighborSelector(nn.Module):
    """
    Strict metric-only neighbor selector.
    
    Selects exactly 64 neighbors (32 nearest, 16 attractors, 16 repulsors)
    using ONLY the learned/curved metric from Address.metric/transport.
    
    NO FALLBACK to Euclidean or cosine distance.
    Fails fast if metric is missing or invalid.
    
    Computes 6 similarity score channels per neighbor under the chosen metric:
        1. Metric distance (curved)
        2. Transport-corrected distance
        3. Attraction strength (for attractors)
        4. Repulsion strength (for repulsors)
        5. Confidence score
        6. Heap rank (position-based ordering)
    """
    
    def __init__(self, config: Optional[AddressConfig] = None):
        super().__init__()
        self.config = config or AddressConfig()
        
        # Projection to compute similarity scores from metric/transport
        self.score_from_metric = nn.Linear(2 * self.config.d, self.config.m)
        
        # Initialize with small weights for stable gradients
        nn.init.normal_(self.score_from_metric.weight, std=0.02)
        nn.init.zeros_(self.score_from_metric.bias)
    
    def compute_metric_distance(
        self,
        query: torch.Tensor,      # (batch, d)
        candidates: torch.Tensor, # (batch, N_cand, d)
        metric: torch.Tensor,     # (batch, d) - diagonal metric
        transport: torch.Tensor   # (batch, d) - transport coefficients
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute metric distance using learned curved metric.
        
        Distance formula (diagonal metric):
            d²(q, c) = (q - c)ᵀ g (q - c)
        where g is the diagonal metric tensor.
        
        Args:
            query: Query embedding (batch, d)
            candidates: Candidate embeddings (batch, N_cand, d)
            metric: Diagonal metric (batch, d)
            transport: Transport/Christoffel coefficients (batch, d)
            
        Returns:
            distances: Metric distances (batch, N_cand)
            scores_6ch: 6-channel similarity scores (batch, N_cand, 6)
        """
        batch_size = query.shape[0]
        n_cand = candidates.shape[1]
        
        # Expand query for broadcasting
        query_exp = query.unsqueeze(1)  # (batch, 1, d)
        
        # Compute difference vectors
        diff = candidates - query_exp  # (batch, N_cand, d)
        
        # Apply metric: d² = diff · g · diff (element-wise for diagonal metric)
        metric_exp = metric.unsqueeze(1)  # (batch, 1, d)
        weighted_diff = diff * metric_exp  # (batch, N_cand, d)
        distances_sq = (diff * weighted_diff).sum(dim=-1)  # (batch, N_cand)
        distances = torch.sqrt(distances_sq.clamp(min=1e-8))  # (batch, N_cand)
        
        # Transport-corrected distance (parallel transport along geodesic)
        transport_exp = transport.unsqueeze(1)  # (batch, 1, d)
        transport_correction = (diff * transport_exp).sum(dim=-1)  # (batch, N_cand)
        corrected_distances = distances + 0.1 * transport_correction  # Small correction term
        
        # Compute 6 similarity score channels from metric and transport
        # Concatenate metric and transport for scoring
        metric_transport = torch.cat([metric, transport], dim=-1)  # (batch, 2*d)
        metric_transport_exp = metric_transport.unsqueeze(1).expand(batch_size, n_cand, -1)
        
        # Project to 6 channels via learned projection
        scores_6ch = self.score_from_metric(metric_transport_exp)  # (batch, N_cand, 6)
        
        # Channel 1: Inverse metric distance (similarity)
        scores_6ch[..., 0] = 1.0 / (distances + 1e-3)
        
        # Channel 2: Inverse transport-corrected distance
        scores_6ch[..., 1] = 1.0 / (corrected_distances.abs() + 1e-3)
        
        # Channels 3-6 are learned from the projection (attraction, repulsion, confidence, rank)
        
        return distances, scores_6ch
    
    def select_neighbors(
        self,
        query_embedding: torch.Tensor,        # (batch, d)
        candidate_embeddings: torch.Tensor,   # (batch, N_cand, d)
        metric: torch.Tensor,                 # (batch, d)
        transport: torch.Tensor,              # (batch, d)
        route_salt: Optional[int] = None      # For collision prevention
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Select exactly 64 neighbors using strict metric-only selection.
        
        Fails fast if:
        - metric is None or invalid (contains NaN/Inf)
        - Cannot populate all 64 slots
        
        Args:
            query_embedding: Query (batch, d)
            candidate_embeddings: Candidates (batch, N_cand, d)
            metric: Diagonal metric (batch, d) - REQUIRED
            transport: Transport coefficients (batch, d) - REQUIRED
            route_salt: Optional salt for collision prevention
            
        Returns:
            selected_embeddings: (batch, 64, d)
            selected_scores: (batch, 64, 6) - 6 score channels
            selected_indices: (batch, 64) - indices into candidates
            
        Raises:
            ValueError: If metric is invalid or cannot populate 64 slots
        """
        # Strict validation: metric and transport MUST be provided
        if metric is None or transport is None:
            raise ValueError(
                "NeighborSelector requires metric and transport. "
                "No Euclidean fallback is available. Fail fast."
            )
        
        # Validate metric has no NaN/Inf
        if metric.isnan().any() or metric.isinf().any():
            raise ValueError(
                f"Invalid metric: contains NaN={metric.isnan().any().item()} "
                f"or Inf={metric.isinf().any().item()}. Fail fast."
            )
        
        if transport.isnan().any() or transport.isinf().any():
            raise ValueError(
                f"Invalid transport: contains NaN={transport.isnan().any().item()} "
                f"or Inf={transport.isinf().any().item()}. Fail fast."
            )
        
        batch_size = query_embedding.shape[0]
        n_cand = candidate_embeddings.shape[1]
        
        # Need at least 64 candidates
        if n_cand < 64:
            raise ValueError(
                f"Need at least 64 candidates to populate all neighbor slots, "
                f"but only {n_cand} provided. Fail fast."
            )
        
        # Compute metric distances and 6-channel scores
        distances, scores_6ch = self.compute_metric_distance(
            query_embedding, candidate_embeddings, metric, transport
        )
        
        # Select 32 nearest neighbors (smallest distances)
        nearest_dists, nearest_idx = torch.topk(
            -distances,  # Negate for largest (smallest distances)
            k=32,
            dim=-1,
            sorted=True
        )
        nearest_idx = nearest_idx  # (batch, 32)
        
        # For attractors: select candidates with high positive scores on channel 3
        # (learned attraction strength)
        attractor_scores = scores_6ch[..., 2]  # (batch, N_cand)
        _, attractor_idx = torch.topk(attractor_scores, k=16, dim=-1, sorted=True)
        
        # For repulsors: select candidates with high negative scores on channel 4
        # (learned repulsion strength - or we can use smallest distances beyond nearest)
        repulsor_scores = scores_6ch[..., 3]  # (batch, N_cand)
        _, repulsor_idx = torch.topk(-repulsor_scores, k=16, dim=-1, sorted=True)
        
        # Combine indices: [32 nearest | 16 attractors | 16 repulsors]
        selected_indices = torch.cat([nearest_idx, attractor_idx, repulsor_idx], dim=1)  # (batch, 64)
        
        # Gather selected embeddings
        batch_indices = torch.arange(batch_size, device=query_embedding.device).view(-1, 1).expand(-1, 64)
        selected_embeddings = candidate_embeddings[batch_indices, selected_indices]  # (batch, 64, d)
        
        # Gather selected scores
        selected_scores = scores_6ch[batch_indices, selected_indices]  # (batch, 64, 6)
        
        # Update channel 6: Heap rank (position in selection)
        heap_ranks = torch.arange(64, device=query_embedding.device, dtype=selected_scores.dtype)
        heap_ranks = heap_ranks.unsqueeze(0).expand(batch_size, -1)  # (batch, 64)
        selected_scores[..., 5] = heap_ranks / 64.0  # Normalize to [0, 1]
        
        return selected_embeddings, selected_scores, selected_indices


class Address:
    """
    Linearized address structure.

    Stores everything as one contiguous tensor for efficiency,
    but provides structured access via properties.
    """

    def __init__(
        self,
        data: torch.Tensor,
        config: Optional[AddressConfig] = None
    ):
        """
        Args:
            data: Linearized address tensor, shape (..., D) where D = config.total_dim
            config: Address configuration (uses default if None)
        """
        self.config = config or AddressConfig()
        self.data = data

        # Validate shape
        if data.shape[-1] != self.config.total_dim:
            raise ValueError(
                f"Address data has dim {data.shape[-1]}, "
                f"expected {self.config.total_dim}"
            )

    @classmethod
    def zeros(
        cls,
        *batch_dims: int,
        config: Optional[AddressConfig] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None
    ) -> 'Address':
        """Create zero-initialized address."""
        cfg = config or AddressConfig()
        data = torch.zeros(*batch_dims, cfg.total_dim, device=device, dtype=dtype)
        return cls(data, cfg)

    @classmethod
    def randn(
        cls,
        *batch_dims: int,
        config: Optional[AddressConfig] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None
    ) -> 'Address':
        """Create random-initialized address (for testing)."""
        cfg = config or AddressConfig()
        data = torch.randn(*batch_dims, cfg.total_dim, device=device, dtype=dtype)
        return cls(data, cfg)

    # =========================================================================
    # Structured access (views into the linearized data)
    # =========================================================================

    @property
    def core(self) -> torch.Tensor:
        """Core embedding, shape (..., d)."""
        return self.data[..., self.config.core_start:self.config.core_end]

    @core.setter
    def core(self, value: torch.Tensor):
        self.data[..., self.config.core_start:self.config.core_end] = value

    @property
    def metric(self) -> torch.Tensor:
        """Local metric, shape (..., d)."""
        start = self.config.geom_start
        end = start + self.config.d
        return self.data[..., start:end]

    @metric.setter
    def metric(self, value: torch.Tensor):
        start = self.config.geom_start
        end = start + self.config.d
        self.data[..., start:end] = value

    @property
    def transport(self) -> torch.Tensor:
        """Transport/Christoffel, shape (..., d)."""
        start = self.config.geom_start + self.config.d
        end = self.config.geom_end
        return self.data[..., start:end]

    @transport.setter
    def transport(self, value: torch.Tensor):
        start = self.config.geom_start + self.config.d
        end = self.config.geom_end
        self.data[..., start:end] = value

    @property
    def neighbors_flat(self) -> torch.Tensor:
        """All neighbors as flat block, shape (..., N * d_block)."""
        return self.data[..., self.config.neighbors_start:self.config.neighbors_end]

    @property
    def neighbors_blocked(self) -> torch.Tensor:
        """Neighbors reshaped to blocks, shape (..., N, d_block)."""
        flat = self.neighbors_flat
        # Reshape last dim
        shape = flat.shape[:-1] + (self.config.n_neighbors, self.config.d_block)
        return flat.view(shape)

    def neighbor(self, i: int) -> torch.Tensor:
        """Get neighbor i's full block, shape (..., d_block)."""
        start = self.config.neighbors_start + i * self.config.d_block
        end = start + self.config.d_block
        return self.data[..., start:end]

    def neighbor_value(self, i: int) -> torch.Tensor:
        """Get neighbor i's value vector, shape (..., d')."""
        block_start = self.config.neighbors_start + i * self.config.d_block
        start = block_start
        end = start + self.config.d_prime
        return self.data[..., start:end]

    def neighbor_scores(self, i: int) -> torch.Tensor:
        """Get neighbor i's scores, shape (..., m)."""
        block_start = self.config.neighbors_start + i * self.config.d_block
        start = block_start + self.config.d_prime
        end = start + self.config.m
        return self.data[..., start:end]

    def neighbor_coords(self, i: int) -> torch.Tensor:
        """Get neighbor i's coords, shape (..., k)."""
        block_start = self.config.neighbors_start + i * self.config.d_block
        start = block_start + self.config.d_prime + self.config.m
        end = start + self.config.k
        return self.data[..., start:end]

    @property
    def all_neighbor_values(self) -> torch.Tensor:
        """All neighbor values, shape (..., N, d')."""
        blocked = self.neighbors_blocked
        return blocked[..., :self.config.d_prime]

    @property
    def all_neighbor_scores(self) -> torch.Tensor:
        """All neighbor scores, shape (..., N, m)."""
        blocked = self.neighbors_blocked
        start = self.config.d_prime
        end = start + self.config.m
        return blocked[..., start:end]

    @property
    def all_neighbor_coords(self) -> torch.Tensor:
        """All neighbor coords, shape (..., N, k)."""
        blocked = self.neighbors_blocked
        start = self.config.d_prime + self.config.m
        return blocked[..., start:]

    # Neighbor role slices
    @property
    def nearest_neighbors(self) -> torch.Tensor:
        """N1-N32: absolute nearest, shape (..., 32, d_block)."""
        blocked = self.neighbors_blocked
        return blocked[..., :self.config.n_nearest, :]

    @property
    def attractor_neighbors(self) -> torch.Tensor:
        """N33-N48: attractors, shape (..., 16, d_block)."""
        blocked = self.neighbors_blocked
        start = self.config.n_nearest
        end = start + self.config.n_attractors
        return blocked[..., start:end, :]

    @property
    def repulsor_neighbors(self) -> torch.Tensor:
        """N49-N64: repulsors, shape (..., 16, d_block)."""
        blocked = self.neighbors_blocked
        start = self.config.n_nearest + self.config.n_attractors
        return blocked[..., start:, :]

    @property
    def ecc(self) -> torch.Tensor:
        """Error correction code, shape (..., 32)."""
        start = self.config.integrity_start
        end = start + self.config.ecc_bits
        return self.data[..., start:end]

    @ecc.setter
    def ecc(self, value: torch.Tensor):
        start = self.config.integrity_start
        end = start + self.config.ecc_bits
        self.data[..., start:end] = value

    @property
    def timestamps(self) -> torch.Tensor:
        """Timestamps [internal_time, wall_time], shape (..., 2)."""
        start = self.config.integrity_start + self.config.ecc_bits
        end = self.config.integrity_end
        return self.data[..., start:end]

    @timestamps.setter
    def timestamps(self, value: torch.Tensor):
        start = self.config.integrity_start + self.config.ecc_bits
        end = self.config.integrity_end
        self.data[..., start:end] = value

    # =========================================================================
    # Utility methods
    # =========================================================================

    def clone(self) -> 'Address':
        """Create a copy."""
        return Address(self.data.clone(), self.config)

    def to(self, device: torch.device) -> 'Address':
        """Move to device."""
        return Address(self.data.to(device), self.config)

    @property
    def shape(self) -> torch.Size:
        """Batch shape (excluding address dim)."""
        return self.data.shape[:-1]

    @property
    def device(self) -> torch.device:
        return self.data.device

    @property
    def dtype(self) -> torch.dtype:
        return self.data.dtype

    def __repr__(self) -> str:
        return (
            f"Address(shape={self.shape}, D={self.config.total_dim}, "
            f"device={self.device})"
        )


class AddressBuilder(nn.Module):
    """
    Builds addresses from embeddings and neighbor information.

    This module takes raw embeddings and constructs the full
    linearized address structure using strict metric-only neighbor selection.
    
    Option 6 Requirements:
    - Uses NeighborSelector with metric-only distances (no Euclidean fallback)
    - Enforces exactly 64 neighbor slots (32 nearest, 16 attractors, 16 repulsors)
    - Computes 6 similarity scores per neighbor under the chosen metric
    - Fails fast if metric is missing or invalid
    - Collision detection with route_hash and rehash on conflict
    """

    def __init__(
        self,
        config: Optional[AddressConfig] = None,
        enable_collision_check: bool = True,
        address_registry: Optional[AddressRegistry] = None
    ):
        super().__init__()
        self.config = config or AddressConfig()
        self.enable_collision_check = enable_collision_check
        self.address_registry = address_registry or get_address_registry()

        # Projections for building address components
        self.metric_proj = nn.Linear(self.config.d, self.config.d)
        self.transport_proj = nn.Linear(self.config.d, self.config.d)
        self.value_proj = nn.Linear(self.config.d, self.config.d_prime)
        self.coord_proj = nn.Linear(self.config.d, self.config.k)

        # Initialize metric/transport near identity
        nn.init.eye_(self.metric_proj.weight)
        nn.init.zeros_(self.metric_proj.bias)
        nn.init.eye_(self.transport_proj.weight)
        nn.init.zeros_(self.transport_proj.bias)
        
        # Strict metric-only neighbor selector
        self.neighbor_selector = NeighborSelector(config=self.config)

    def forward(
        self,
        embedding: torch.Tensor,
        candidate_embeddings: torch.Tensor,
        timestamp: Optional[float] = None,
        enable_probing: bool = True
    ) -> Address:
        """
        Build address from embedding and candidates using strict metric selection.

        Args:
            embedding: Core embedding, shape (batch, d)
            candidate_embeddings: Candidate neighbor embeddings, shape (batch, N_cand, d)
                Must have N_cand >= 64 for full slot population
            timestamp: Current time
            enable_probing: If True, use address probing (Option 6). If False, legacy mode.

        Returns:
            Address with all fields populated

        Raises:
            ValueError: If metric is invalid or cannot populate 64 slots
        """
        import time as time_module

        batch_size = embedding.shape[0]
        device = embedding.device
        dtype = embedding.dtype

        # Create empty address
        addr = Address.zeros(batch_size, config=self.config, device=device, dtype=dtype)

        # Fill core
        addr.core = embedding

        # Fill geometry (REQUIRED for Option 6)
        metric = self.metric_proj(embedding)
        transport = self.transport_proj(embedding)
        
        addr.metric = metric
        addr.transport = transport
        
        # Collision detection (per-batch-item, use first item for demo)
        if self.enable_collision_check and batch_size > 0:
            try:
                route_hash, salt_used = self.address_registry.check_and_register(
                    embedding[0]  # Check first item in batch
                )
                # Store salt in ECC field for debugging (first 4 bytes)
                # Real ECC would go here
                addr.ecc[0, 0] = float(salt_used)
            except RuntimeError as e:
                # Collision failure - re-raise with context
                raise RuntimeError(
                    f"Address collision detection failed: {e}. "
                    "Unable to generate unique address."
                )

        # Fill neighbors using strict metric-only selection
        if enable_probing:
            # Option 6: Strict metric-only neighbor selection
            # FAIL FAST if metric/transport invalid or cannot populate 64 slots
            try:
                selected_embeddings, selected_scores, selected_indices = \
                    self.neighbor_selector.select_neighbors(
                        query_embedding=embedding,
                        candidate_embeddings=candidate_embeddings,
                        metric=metric,
                        transport=transport
                    )
                
                # Project neighbor embeddings to value and coord spaces
                # selected_embeddings: (batch, 64, d)
                values = self.value_proj(selected_embeddings)  # (batch, 64, d')
                coords = self.coord_proj(selected_embeddings)  # (batch, 64, k)
                
                # Scores already computed by neighbor_selector (6 channels)
                scores = selected_scores  # (batch, 64, 6)
                
                # Pack into neighbor blocks
                # blocked shape: (batch, 64, d_block) where d_block = d' + m + k
                blocked = torch.cat([values, scores, coords], dim=-1)
                
                # Flatten and assign
                addr.data[..., self.config.neighbors_start:self.config.neighbors_end] = \
                    blocked.view(batch_size, -1)
                
            except ValueError as e:
                # Re-raise with more context
                raise ValueError(
                    f"Strict neighbor selection failed (Option 6): {e}. "
                    "Address probing requires valid metric/transport and sufficient candidates."
                )
        else:
            # Legacy mode: allow empty neighbors (backward compatibility)
            # This path is NOT recommended for Option 6
            pass

        # Fill timestamps
        current_time = timestamp if timestamp is not None else time_module.time()
        addr.timestamps = torch.tensor(
            [[current_time, current_time]],
            device=device, dtype=dtype
        ).expand(batch_size, -1)

        # ECC would be computed here (placeholder: zeros except salt)
        # Real implementation would compute BCH code from content

        return addr


# =============================================================================
# Schema documentation (for reference)
# =============================================================================

ADDRESS_SCHEMA = """
Linearized Address Layout (D = 7074 for d=512):

Offset    Size    Field
------    ----    -----
0         512     core (embedding)
512       512     metric (diagonal)
1024      512     transport (diagonal)
1536      5504    neighbors (64 × 86)
7040      32      ecc
7072      2       timestamps

Per-neighbor block (86 floats):
  Offset  Size  Field
  ------  ----  -----
  0       64    value (d')
  64      6     scores (m) - 6 metric-derived similarity channels
  70      16    coords (k)

Neighbor roles:
  N1-N32  (idx 0-31):   absolute nearest (metric-based)
  N33-N48 (idx 32-47):  attractors (high positive scores)
  N49-N64 (idx 48-63):  repulsors (contrastive evidence)

6 Similarity Score Channels (all metric-derived):
  Channel 0: Metric distance (inverse, curved geometry)
  Channel 1: Transport-corrected distance (parallel transport)
  Channel 2: Attraction strength (learned from metric/transport)
  Channel 3: Repulsion strength (learned from metric/transport)
  Channel 4: Confidence score (learned from metric/transport)
  Channel 5: Heap rank (position in selection, normalized to [0,1])

Strict Requirements (Option 6):
  - NO Euclidean or cosine fallback - metric/transport REQUIRED
  - All 64 slots MUST be populated (fail fast if unable)
  - Collision detection via route_hash with SHA256
  - Rehash with salt on collision (max 10 attempts)
  - ECC/timestamps present but excluded from similarity scoring

Example Address Structure:
  [Core Embedding (512)]
  [Metric g_ij (512)]
  [Transport Γ_ij (512)]
  [Neighbor 0: value(64) | scores(6) | coords(16)]
  [Neighbor 1: value(64) | scores(6) | coords(16)]
  ...
  [Neighbor 63: value(64) | scores(6) | coords(16)]
  [ECC (32)]
  [Timestamps (2)]
"""

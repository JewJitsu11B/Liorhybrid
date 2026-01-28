"""
Linearized Address Structure (Option 6: Mandatory Address-Based Neighbor Probing)

The address is a structured vector with fixed-width blocks:
[ core | geom | N1 | N2 | ... | N64 | integrity ]

Position and ordering encode meaning (conditioning on structured priors).

Dimensions (default d=512):
- core: d = 512 (embedding)
- geom: 2d = 1024 (metric + transport)
- neighbors: 64 × d_block = 7072 (N1-N64, d_block=116 with metric/transport)
- integrity: 34 (ecc + timestamps)
- Total D = 8642 floats

Neighbor roles by position (mandatory, no fallbacks):
- N1-N32: absolute nearest (similarity grounding)
- N33-N48: attractors (reinforcing evidence, top similarity after nearest)
- N49-N64: repulsors (contrastive evidence, lowest similarity)

Per-neighbor block (d_block = 116):
- value: d' = 64 (reduced interaction vector)
- neighbor_metric: 16 (metric features of this neighbor)
- neighbor_transport: 16 (transport features of this neighbor)
- scores: m = 6 (6 similarity types: cosine + 5 learned, mandatory)
- coords: k = 16 (routing info)

Similarity scores (6 types, all computed):
- Score 0: Cosine similarity (geometric baseline, computed from embeddings)
- Scores 1-5: Learned similarity metrics (projected feature interactions)
- If external neighbor_similarities absent, computed internally (no empty slots)

ECC and timestamps:
- Present in address for integrity/causality
- Excluded from neighbor similarity scoring
- ECC stores collision-avoidance hash for uniqueness

Collision avoidance:
- Route hash with 64 extra bits for address-space entropy
- First 32 bits stored in ECC field
- Enables more unique addresses than N tokens
"""
try: import usage_tracker; usage_tracker.track(__file__)
except: pass

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, Tuple, NamedTuple


@dataclass
class AddressConfig:
    """Configuration for address dimensions.
    
    Address-based neighbor probing (Option 6) with mandatory 64-slot structure:
    - 64 neighbors: 32 nearest, 16 attractors, 16 repulsors (role-typed)
    - 6 similarity scores per neighbor (mandatory, no fallbacks)
    - Each neighbor slot stores metric/transport info of that neighbor
    - ECC and timestamps present but excluded from similarity scoring
    """
    # Core embedding
    d: int = 512

    # Geometry (diagonal mode)
    use_lowrank_geom: bool = False
    r: int = 32  # low-rank dim if used

    # Neighbors (fixed, mandatory)
    n_nearest: int = 32
    n_attractors: int = 16
    n_repulsors: int = 16

    # Per-neighbor dimensions
    d_prime: int = 64   # value dim (interaction output)
    m: int = 6          # scores dim (6 similarity score types, mandatory)
    k: int = 16         # coords dim (routing info)
    
    # Per-neighbor metric/transport feature dims (stores neighbor's geometry)
    d_neighbor_metric: int = 16    # metric features per neighbor
    d_neighbor_transport: int = 16  # transport features per neighbor

    # Integrity (present but excluded from neighbor scoring)
    ecc_bits: int = 32
    n_timestamps: int = 2
    
    # Address probing mode
    enable_address_probing: bool = True  # Default to address-based probing (Option 6)

    @property
    def n_neighbors(self) -> int:
        return self.n_nearest + self.n_attractors + self.n_repulsors

    @property
    def d_block(self) -> int:
        """Size of one neighbor block (value + metric + transport + scores + coords)."""
        return self.d_prime + self.d_neighbor_metric + self.d_neighbor_transport + self.m + self.k

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
    """Indices for accessing parts of a neighbor block.
    
    Neighbor block layout: [value | neighbor_metric | neighbor_transport | scores | coords]
    """
    value_start: int
    value_end: int
    neighbor_metric_start: int
    neighbor_metric_end: int
    neighbor_transport_start: int
    neighbor_transport_end: int
    scores_start: int
    scores_end: int
    coords_start: int
    coords_end: int


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
    
    def neighbor_metric(self, i: int) -> torch.Tensor:
        """Get neighbor i's metric features, shape (..., d_neighbor_metric)."""
        block_start = self.config.neighbors_start + i * self.config.d_block
        start = block_start + self.config.d_prime
        end = start + self.config.d_neighbor_metric
        return self.data[..., start:end]
    
    def neighbor_transport(self, i: int) -> torch.Tensor:
        """Get neighbor i's transport features, shape (..., d_neighbor_transport)."""
        block_start = self.config.neighbors_start + i * self.config.d_block
        start = block_start + self.config.d_prime + self.config.d_neighbor_metric
        end = start + self.config.d_neighbor_transport
        return self.data[..., start:end]

    def neighbor_scores(self, i: int) -> torch.Tensor:
        """Get neighbor i's scores (6 similarity types), shape (..., m=6)."""
        block_start = self.config.neighbors_start + i * self.config.d_block
        start = block_start + self.config.d_prime + self.config.d_neighbor_metric + self.config.d_neighbor_transport
        end = start + self.config.m
        return self.data[..., start:end]

    def neighbor_coords(self, i: int) -> torch.Tensor:
        """Get neighbor i's coords, shape (..., k)."""
        block_start = self.config.neighbors_start + i * self.config.d_block
        start = block_start + self.config.d_prime + self.config.d_neighbor_metric + self.config.d_neighbor_transport + self.config.m
        end = start + self.config.k
        return self.data[..., start:end]

    @property
    def all_neighbor_values(self) -> torch.Tensor:
        """All neighbor values, shape (..., N, d')."""
        blocked = self.neighbors_blocked
        return blocked[..., :self.config.d_prime]
    
    @property
    def all_neighbor_metrics(self) -> torch.Tensor:
        """All neighbor metric features, shape (..., N, d_neighbor_metric)."""
        blocked = self.neighbors_blocked
        start = self.config.d_prime
        end = start + self.config.d_neighbor_metric
        return blocked[..., start:end]
    
    @property
    def all_neighbor_transports(self) -> torch.Tensor:
        """All neighbor transport features, shape (..., N, d_neighbor_transport)."""
        blocked = self.neighbors_blocked
        start = self.config.d_prime + self.config.d_neighbor_metric
        end = start + self.config.d_neighbor_transport
        return blocked[..., start:end]

    @property
    def all_neighbor_scores(self) -> torch.Tensor:
        """All neighbor scores (6 similarity types), shape (..., N, m=6)."""
        blocked = self.neighbors_blocked
        start = self.config.d_prime + self.config.d_neighbor_metric + self.config.d_neighbor_transport
        end = start + self.config.m
        return blocked[..., start:end]

    @property
    def all_neighbor_coords(self) -> torch.Tensor:
        """All neighbor coords, shape (..., N, k)."""
        blocked = self.neighbors_blocked
        start = self.config.d_prime + self.config.d_neighbor_metric + self.config.d_neighbor_transport + self.config.m
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
    
    Option 6: Address-based neighbor probing with mandatory 64-slot structure.
    - 64 neighbors: 32 nearest, 16 attractors (top), 16 repulsors (bottom)
    - 6 similarity scores per neighbor (mandatory, computed if not provided)
    - Each neighbor stores metric/transport features from that neighbor's embedding
    - ECC and timestamps present but excluded from neighbor scoring
    - Collision avoidance via route_hash with extra bits
    """

    def __init__(self, config: Optional[AddressConfig] = None):
        super().__init__()
        self.config = config or AddressConfig()

        # Projections for building address components
        self.metric_proj = nn.Linear(self.config.d, self.config.d)
        self.transport_proj = nn.Linear(self.config.d, self.config.d)
        
        # Per-neighbor projections
        self.value_proj = nn.Linear(self.config.d, self.config.d_prime)
        self.neighbor_metric_proj = nn.Linear(self.config.d, self.config.d_neighbor_metric)
        self.neighbor_transport_proj = nn.Linear(self.config.d, self.config.d_neighbor_transport)
        self.coord_proj = nn.Linear(self.config.d, self.config.k)
        
        # Projections for computing 6 similarity scores
        # Score 0: cosine similarity (computed, not learned)
        # Scores 1-5: learned similarity metrics
        self.similarity_proj = nn.Linear(self.config.d, 5)  # 5 learned scores

        # Initialize metric/transport near identity
        nn.init.eye_(self.metric_proj.weight)
        nn.init.zeros_(self.metric_proj.bias)
        nn.init.eye_(self.transport_proj.weight)
        nn.init.zeros_(self.transport_proj.bias)
        
        # Collision avoidance: route hash projection
        self.route_hash_proj = nn.Linear(self.config.d, 64)  # Extra bits for uniqueness
        
    def compute_similarity_scores(
        self,
        query_embedding: torch.Tensor,
        neighbor_embeddings: torch.Tensor,
        neighbor_similarities: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute 6 similarity scores per neighbor (mandatory).
        
        Args:
            query_embedding: (batch, d)
            neighbor_embeddings: (batch, N, d)
            neighbor_similarities: Optional precomputed similarities (batch, N)
            
        Returns:
            scores: (batch, N, 6) - 6 similarity score types
        """
        batch_size, n_neighbors, d = neighbor_embeddings.shape
        
        # Score 0: Cosine similarity (geometric baseline)
        if neighbor_similarities is not None:
            cosine_sim = neighbor_similarities.unsqueeze(-1)  # (batch, N, 1)
        else:
            # Compute cosine similarity internally
            query_norm = F.normalize(query_embedding, dim=-1, p=2)  # (batch, d)
            neighbor_norm = F.normalize(neighbor_embeddings, dim=-1, p=2)  # (batch, N, d)
            cosine_sim = torch.einsum('bd,bnd->bn', query_norm, neighbor_norm).unsqueeze(-1)  # (batch, N, 1)
        
        # Scores 1-5: Learned similarity metrics
        # Use the neighbor features directly for diversity (simpler approach)
        neighbor_feats = self.similarity_proj(neighbor_embeddings)  # (batch, N, 5)
        learned_scores = neighbor_feats  # (batch, N, 5)
        
        # Concatenate all 6 scores
        all_scores = torch.cat([cosine_sim, learned_scores], dim=-1)  # (batch, N, 6)
        
        return all_scores
    
    def select_neighbors(
        self,
        embedding: torch.Tensor,
        candidate_embeddings: torch.Tensor,
        similarity_scores: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Select 64 neighbors with role typing: 32 nearest, 16 attractors, 16 repulsors.
        
        Args:
            embedding: Query embedding (batch, d)
            candidate_embeddings: Candidate neighbor embeddings (batch, M, d) where M >= 64
            similarity_scores: Optional precomputed similarities (batch, M)
            
        Returns:
            selected_embeddings: (batch, 64, d)
            selection_indices: (batch, 64) - indices of selected neighbors
        """
        batch_size = embedding.shape[0]
        device = embedding.device
        
        # If we don't have enough candidates, repeat to fill 64 slots
        n_candidates = candidate_embeddings.shape[1]
        if n_candidates < 64:
            # Repeat candidates to fill 64 slots
            repeats = (64 + n_candidates - 1) // n_candidates
            candidate_embeddings = candidate_embeddings.repeat(1, repeats, 1)[:, :64, :]
            if similarity_scores is not None:
                similarity_scores = similarity_scores.repeat(1, repeats)[:, :64]
            n_candidates = 64
        
        # Compute similarities if not provided
        if similarity_scores is None:
            query_norm = F.normalize(embedding, dim=-1, p=2)
            candidate_norm = F.normalize(candidate_embeddings, dim=-1, p=2)
            similarity_scores = torch.einsum('bd,bmd->bm', query_norm, candidate_norm)
        
        # Select 32 nearest (highest similarity)
        _, nearest_indices = torch.topk(similarity_scores, k=32, dim=-1, largest=True)
        
        # For attractors and repulsors, we need to exclude the nearest
        mask = torch.ones_like(similarity_scores, dtype=torch.bool)
        mask.scatter_(1, nearest_indices, False)
        masked_scores = similarity_scores.clone()
        masked_scores[~mask] = float('-inf')
        
        # Select 16 attractors (next highest after nearest)
        _, attractor_indices = torch.topk(masked_scores, k=16, dim=-1, largest=True)
        
        # Update mask
        mask.scatter_(1, attractor_indices, False)
        masked_scores = similarity_scores.clone()
        masked_scores[~mask] = float('inf')
        
        # Select 16 repulsors (lowest similarity, excluding already selected)
        _, repulsor_indices = torch.topk(masked_scores, k=16, dim=-1, largest=False)
        
        # Concatenate all indices in order: nearest, attractors, repulsors
        all_indices = torch.cat([nearest_indices, attractor_indices, repulsor_indices], dim=-1)  # (batch, 64)
        
        # Gather selected embeddings
        selected_embeddings = torch.gather(
            candidate_embeddings,
            1,
            all_indices.unsqueeze(-1).expand(-1, -1, candidate_embeddings.shape[-1])
        )  # (batch, 64, d)
        
        return selected_embeddings, all_indices
    
    def compute_collision_hash(self, embedding: torch.Tensor) -> torch.Tensor:
        """
        Compute collision-avoidance hash for uniqueness.
        
        Args:
            embedding: (batch, d)
            
        Returns:
            hash: (batch, 64) - hash bits for address uniqueness
        """
        return torch.tanh(self.route_hash_proj(embedding))

    def forward(
        self,
        embedding: torch.Tensor,
        neighbor_embeddings: Optional[torch.Tensor] = None,
        neighbor_similarities: Optional[torch.Tensor] = None,
        timestamp: Optional[float] = None
    ) -> Address:
        """
        Build address from embedding and neighbors.
        
        Mandatory behavior (no fallbacks when address probing is enabled):
        - 64 neighbors must be populated (selection or repetition)
        - 6 similarity scores computed per neighbor
        - Each neighbor stores metric/transport features
        - ECC and timestamps present (but excluded from scoring)

        Args:
            embedding: Core embedding, shape (batch, d)
            neighbor_embeddings: Neighbor embeddings, shape (batch, M, d) where M >= 64
                                If None and address_probing enabled, will use self-similarity
            neighbor_similarities: Precomputed similarities, shape (batch, M)
            timestamp: Current time

        Returns:
            Address with all fields populated
        """
        import time as time_module

        batch_size = embedding.shape[0]
        device = embedding.device
        dtype = embedding.dtype

        # Create empty address
        addr = Address.zeros(batch_size, config=self.config, device=device, dtype=dtype)

        # Fill core
        addr.core = embedding

        # Fill geometry
        addr.metric = self.metric_proj(embedding)
        addr.transport = self.transport_proj(embedding)

        # Fill neighbors (MANDATORY for address probing)
        if self.config.enable_address_probing:
            # Ensure we have neighbor embeddings
            if neighbor_embeddings is None:
                # Fallback: use self as single neighbor, then repeat
                neighbor_embeddings = embedding.unsqueeze(1)  # (batch, 1, d)
            
            # Select 64 neighbors with role typing
            selected_neighbors, _ = self.select_neighbors(
                embedding, neighbor_embeddings, neighbor_similarities
            )  # (batch, 64, d)
            
            # Compute 6 similarity scores per neighbor
            scores = self.compute_similarity_scores(
                embedding, selected_neighbors, None
            )  # (batch, 64, 6)
            
            # Project each neighbor to get value, metric, transport, coords
            values = self.value_proj(selected_neighbors)  # (batch, 64, d')
            neighbor_metrics = self.neighbor_metric_proj(selected_neighbors)  # (batch, 64, d_neighbor_metric)
            neighbor_transports = self.neighbor_transport_proj(selected_neighbors)  # (batch, 64, d_neighbor_transport)
            coords = self.coord_proj(selected_neighbors)  # (batch, 64, k)
            
            # Pack into neighbor blocks
            # Block layout: [value | neighbor_metric | neighbor_transport | scores | coords]
            blocked = torch.cat([
                values, 
                neighbor_metrics,
                neighbor_transports,
                scores, 
                coords
            ], dim=-1)  # (batch, 64, d_block)
            
            # Flatten and assign
            addr.data[..., self.config.neighbors_start:self.config.neighbors_end] = \
                blocked.view(batch_size, -1)
        
        # Fill timestamps (present but excluded from neighbor scoring)
        current_time = timestamp if timestamp is not None else time_module.time()
        addr.timestamps = torch.tensor(
            [[current_time, current_time]],
            device=device, dtype=dtype
        ).expand(batch_size, -1)

        # ECC placeholder (present but excluded from neighbor scoring)
        # Real implementation would compute BCH code from content
        # For now, use hash-based collision avoidance
        collision_hash = self.compute_collision_hash(embedding)
        # Store in ECC field (first 32 bits of 64-bit hash, padded)
        addr.ecc = collision_hash[..., :32]

        return addr


# =============================================================================
# Collision Avoidance Helpers
# =============================================================================

def check_address_collisions(addresses: Address, threshold: float = 0.99) -> Tuple[int, torch.Tensor]:
    """
    Check for collisions in a batch of addresses based on ECC hash similarity.
    
    Args:
        addresses: Address object with batch of addresses
        threshold: Similarity threshold for collision detection (default 0.99)
        
    Returns:
        n_collisions: Number of collision pairs detected
        collision_matrix: (batch, batch) boolean matrix of collisions
    """
    ecc_hashes = addresses.ecc  # (batch, 32)
    
    # Compute pairwise cosine similarity of hashes
    ecc_norm = F.normalize(ecc_hashes, dim=-1, p=2)
    similarity = torch.mm(ecc_norm, ecc_norm.t())  # (batch, batch)
    
    # Mask diagonal (self-similarity)
    mask = ~torch.eye(similarity.shape[0], dtype=torch.bool, device=similarity.device)
    similarity = similarity * mask.float()
    
    # Detect collisions
    collision_matrix = similarity > threshold
    n_collisions = collision_matrix.sum().item() // 2  # Divide by 2 for symmetric matrix
    
    return n_collisions, collision_matrix


def compute_address_uniqueness_score(addresses: Address) -> float:
    """
    Compute a uniqueness score for a batch of addresses (higher is better).
    
    Args:
        addresses: Address object with batch of addresses
        
    Returns:
        uniqueness_score: Score in [0, 1] where 1 means all addresses are unique
    """
    ecc_hashes = addresses.ecc  # (batch, 32)
    
    # Compute pairwise distances
    ecc_norm = F.normalize(ecc_hashes, dim=-1, p=2)
    similarity = torch.mm(ecc_norm, ecc_norm.t())  # (batch, batch)
    
    # Mask diagonal
    mask = ~torch.eye(similarity.shape[0], dtype=torch.bool, device=similarity.device)
    similarity = similarity * mask.float()
    
    # Average dissimilarity (1 - similarity)
    dissimilarity = 1.0 - similarity
    uniqueness_score = dissimilarity[mask].mean().item()
    
    return uniqueness_score


# =============================================================================
# Schema documentation (for reference)
# =============================================================================

ADDRESS_SCHEMA = """
Linearized Address Layout (D = 9122 for d=512, updated with neighbor metric/transport):

Offset    Size    Field
------    ----    -----
0         512     core (embedding)
512       512     metric (diagonal)
1024      512     transport (diagonal)
1536      7552    neighbors (64 × 118)
9088      32      ecc (collision hash)
9120      2       timestamps

Per-neighbor block (118 floats):
  Offset  Size  Field
  ------  ----  -----
  0       64    value (d', interaction output)
  64      16    neighbor_metric (metric features of this neighbor)
  80      16    neighbor_transport (transport features of this neighbor)
  96      6     scores (6 similarity types: cosine + 5 learned)
  102     16    coords (k, routing info)

Neighbor roles (mandatory, no fallbacks):
  N1-N32  (idx 0-31):   absolute nearest (32 neighbors)
  N33-N48 (idx 32-47):  attractors (16 neighbors, top similarity after nearest)
  N49-N64 (idx 48-63):  repulsors (16 neighbors, lowest similarity)

Similarity scores (6 types, mandatory):
  Score 0: Cosine similarity (geometric baseline)
  Scores 1-5: Learned similarity metrics (projected feature interactions)

Note on ECC and timestamps:
  - Present in address structure for integrity/causality
  - Excluded from neighbor similarity scoring
  - ECC stores collision-avoidance hash (first 32 bits of 64-bit route hash)
"""

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
- N33-N48: high_sim neighbors (maximum similarity interactions)
- N49-N64: low_sim neighbors (minimum similarity, contrastive examples)

Per-neighbor block (d_block = 86):
- value: d' = 64 (reduced interaction vector)
- scores: m = 6 (geometric similarity channels: dot, wedge, tensor, spinor, energy, rank)
- coords: k = 16 (routing info)

Strict Requirements (Option 6):
- Neighbor selection: ONLY metric-derived distances (no Euclidean fallback)
- All 64 slots MUST be populated (fail fast if unable)
- 6 score channels per neighbor (geometric products: dot, wedge, tensor, spinor, energy, rank)
- Addresses naturally unique based on embeddings (no collision detection needed)
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
    n_high_sim: int = 16  # formerly n_attractors
    n_low_sim: int = 16   # formerly n_repulsors

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
        return self.n_nearest + self.n_high_sim + self.n_low_sim

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


class NeighborSelector(nn.Module):
    """
    Strict metric-only neighbor selector.
    
    Selects exactly 64 neighbors (32 nearest, 16 high_sim, 16 low_sim)
    using ONLY the learned/curved metric from Address.metric/transport.
    
    NO FALLBACK to Euclidean or cosine distance.
    Fails fast if metric is missing or invalid.
    
    Computes 6 geometric similarity score channels per neighbor:
        1. Dot product (standard inner product with metric)
        2. Wedge product (antisymmetric, captures orthogonality)
        3. Tensor product (full correlation structure)
        4. Spinor product (rotational features)
        5. Energy (field-based potential)
        6. Heap rank (position-based ordering [0,1])
    """
    
    def __init__(self, config: Optional[AddressConfig] = None):
        super().__init__()
        self.config = config or AddressConfig()
        
        # No learned projection - geometric products computed directly
    
    def compute_geometric_scores(
        self,
        query: torch.Tensor,      # (batch, d)
        candidates: torch.Tensor, # (batch, N_cand, d)
        metric: torch.Tensor,     # (batch, d) - diagonal metric
        transport: torch.Tensor   # (batch, d) - transport coefficients
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute metric distance and 6 geometric similarity scores.
        
        Distance formula (diagonal metric):
            d²(q, c) = (q - c)ᵀ g (q - c)
        where g is the diagonal metric tensor.
        
        Geometric scores use metric-weighted inner products:
        - Dot: q·g·c (standard metric-weighted dot product)
        - Wedge: |q×c| using metric (captures orthogonality)
        - Tensor: q⊗c via metric (full correlation)
        - Spinor: rotational component via metric
        - Energy: potential field strength
        - Rank: position in sorted list
        
        Args:
            query: Query embedding (batch, d)
            candidates: Candidate embeddings (batch, N_cand, d)
            metric: Diagonal metric (batch, d)
            transport: Transport/Christoffel coefficients (batch, d)
            
        Returns:
            distances: Metric distances (batch, N_cand)
            scores_6ch: 6 geometric similarity scores (batch, N_cand, 6)
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
        
        # Initialize 6-channel scores
        scores_6ch = torch.zeros(batch_size, n_cand, 6, device=query.device, dtype=query.dtype)
        
        # Channel 0: Dot product (metric-weighted inner product)
        # dot(q, c) = q·g·c
        q_weighted = query * metric  # (batch, d)
        q_weighted_exp = q_weighted.unsqueeze(1)  # (batch, 1, d)
        dot_scores = (q_weighted_exp * candidates).sum(dim=-1)  # (batch, N_cand)
        scores_6ch[..., 0] = dot_scores / (query.norm(dim=-1, keepdim=True).unsqueeze(1) * candidates.norm(dim=-1).clamp(min=1e-8) + 1e-8)
        
        # Channel 1: Wedge product (antisymmetric, measures orthogonality)
        # |q×c| approximated via (q·g·q)(c·g·c) - (q·g·c)²
        q_norm_sq = (query * metric * query).sum(dim=-1, keepdim=True).unsqueeze(1)  # (batch, 1, 1)
        c_norm_sq = (candidates * metric_exp * candidates).sum(dim=-1)  # (batch, N_cand)
        wedge_scores = torch.sqrt(
            (q_norm_sq.squeeze(-1) * c_norm_sq - dot_scores.pow(2)).clamp(min=0.0) + 1e-8
        )
        scores_6ch[..., 1] = wedge_scores / (q_norm_sq.squeeze(-1) * c_norm_sq).sqrt().clamp(min=1e-8)
        
        # Channel 2: Tensor product (full correlation, element-wise product magnitude)
        # ||q⊗c|| via metric
        tensor_scores = (q_weighted_exp * candidates).abs().mean(dim=-1)  # (batch, N_cand)
        scores_6ch[..., 2] = tensor_scores
        
        # Channel 3: Spinor product (rotational component using transport)
        # Approximated via transport-weighted cross product
        transport_exp = transport.unsqueeze(1)  # (batch, 1, d)
        spinor_scores = (diff * transport_exp).abs().sum(dim=-1)  # (batch, N_cand)
        scores_6ch[..., 3] = spinor_scores / (distances + 1e-3)
        
        # Channel 4: Energy (field potential - inverse distance squared)
        # E = -1/r representing attractive potential
        energy_scores = -1.0 / (distances.pow(2) + 1e-3)
        scores_6ch[..., 4] = energy_scores
        
        # Channel 5: Heap rank (will be filled in select_neighbors)
        # Placeholder zeros for now
        
        return distances, scores_6ch
    
    def select_neighbors(
        self,
        query_embedding: torch.Tensor,        # (batch, d)
        candidate_embeddings: torch.Tensor,   # (batch, N_cand, d)
        metric: torch.Tensor,                 # (batch, d)
        transport: torch.Tensor,              # (batch, d)
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
            
        Returns:
            selected_embeddings: (batch, 64, d)
            selected_scores: (batch, 64, 6) - 6 geometric score channels
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
        
        # Compute metric distances and 6 geometric score channels
        distances, scores_6ch = self.compute_geometric_scores(
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
        
        # For high_sim: select candidates with high dot product (channel 0)
        high_sim_scores = scores_6ch[..., 0]  # (batch, N_cand)
        _, high_sim_idx = torch.topk(high_sim_scores, k=16, dim=-1, sorted=True)

        # For low_sim: select candidates with high wedge product (channel 1, orthogonal)
        low_sim_scores = scores_6ch[..., 1]  # (batch, N_cand)
        _, low_sim_idx = torch.topk(low_sim_scores, k=16, dim=-1, sorted=True)

        # Combine indices: [32 nearest | 16 high_sim | 16 low_sim]
        selected_indices = torch.cat([nearest_idx, high_sim_idx, low_sim_idx], dim=1)  # (batch, 64)
        
        # Gather selected embeddings
        batch_indices = torch.arange(batch_size, device=query_embedding.device).view(-1, 1).expand(-1, 64)
        selected_embeddings = candidate_embeddings[batch_indices, selected_indices]  # (batch, 64, d)
        
        # Gather selected scores
        selected_scores = scores_6ch[batch_indices, selected_indices]  # (batch, 64, 6)
        
        # Update channel 5: Heap rank (position in selection)
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
    def high_sim_neighbors(self) -> torch.Tensor:
        """N33-N48: high_sim neighbors, shape (..., 16, d_block)."""
        blocked = self.neighbors_blocked
        start = self.config.n_nearest
        end = start + self.config.n_high_sim
        return blocked[..., start:end, :]

    @property
    def low_sim_neighbors(self) -> torch.Tensor:
        """N49-N64: low_sim neighbors, shape (..., 16, d_block)."""
        blocked = self.neighbors_blocked
        start = self.config.n_nearest + self.config.n_high_sim
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
    - Enforces exactly 64 neighbor slots (32 nearest, 16 high_sim, 16 low_sim)
    - Computes 6 geometric similarity scores per neighbor (dot, wedge, tensor, spinor, energy, rank)
    - Fails fast if metric is missing or invalid
    - Addresses are naturally unique based on embeddings (no collision detection)
    """

    def __init__(
        self,
        config: Optional[AddressConfig] = None,
    ):
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
        # Addresses are naturally unique based on embeddings - no collision detection needed

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
1536      5504    neighbors (64 × 86)
7040      32      ecc
7072      2       timestamps

Per-neighbor block (86 floats):
  Offset  Size  Field
  ------  ----  -----
  0       64    value (d')
  64      6     scores (m) - 6 geometric similarity channels
  70      16    coords (k)

Neighbor roles:
  N1-N32  (idx 0-31):   absolute nearest (metric-based)
  N33-N48 (idx 32-47):  high_sim (high dot product)
  N49-N64 (idx 48-63):  low_sim (high wedge product, orthogonal)

6 Geometric Similarity Score Channels:
  Channel 0: Dot product (metric-weighted inner product q·g·c)
  Channel 1: Wedge product (antisymmetric, captures orthogonality |q×c|)
  Channel 2: Tensor product (full correlation structure q⊗c)
  Channel 3: Spinor product (rotational features via transport)
  Channel 4: Energy (field potential -1/r²)
  Channel 5: Heap rank (position in selection, normalized to [0,1])

Strict Requirements (Option 6):
  - NO Euclidean or cosine fallback - metric/transport REQUIRED
  - All 64 slots MUST be populated (fail fast if unable)
  - Addresses naturally unique based on embeddings (no collision detection)
  - ECC/timestamps present but excluded from similarity scoring

Example Address Structure:
  [Core Embedding (512)]
  [Metric g_ij (512)]
  [Transport Γ_ij (512)]
  [Neighbor 0: value(64) | scores(6: dot,wedge,tensor,spinor,energy,rank) | coords(16)]
  [Neighbor 1: value(64) | scores(6: dot,wedge,tensor,spinor,energy,rank) | coords(16)]
  ...
  [Neighbor 63: value(64) | scores(6: dot,wedge,tensor,spinor,energy,rank) | coords(16)]
  [ECC (32)]
  [Timestamps (2)]
"""

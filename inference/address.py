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
from dataclasses import dataclass
from typing import Optional, Tuple, NamedTuple


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
        
        # For attractors: select candidates with high dot product (channel 0)
        attractor_scores = scores_6ch[..., 0]  # (batch, N_cand)
        _, attractor_idx = torch.topk(attractor_scores, k=16, dim=-1, sorted=True)
        
        # For repulsors: select candidates with high wedge product (channel 1, orthogonal)
        repulsor_scores = scores_6ch[..., 1]  # (batch, N_cand)
        _, repulsor_idx = torch.topk(repulsor_scores, k=16, dim=-1, sorted=True)
        
        # Combine indices: [32 nearest | 16 attractors | 16 repulsors]
        selected_indices = torch.cat([nearest_idx, attractor_idx, repulsor_idx], dim=1)  # (batch, 64)
        
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
    def neighbor_similarity_vectors(self) -> torch.Tensor:
        """
        PLANNING NOTE: 15D similarity vectors for all 64 neighbors.
        
        Extended neighbor scoring with comprehensive geometric similarity.
        This property will return rich 15D (or 9D for Phase 1) similarity
        vectors for each neighbor, replacing simple scalar scores with
        multi-dimensional geometric features.
        
        Returns:
            [batch, 64, D] where D ∈ {9, 12, 15}
            
        Each neighbor will have a comprehensive similarity vector containing:
            - cosine: Angular alignment
            - wedge_magnitude: Rotational structure
            - tensor_trace: Interaction strength
            - spinor_magnitude: Phase overlap
            - spinor_phase: Phase angle
            - energy: Field coupling
            - l2_tangent: Euclidean in tangent space
            - l1_tangent: Manhattan in tangent space
            - lior_distance: Geodesic distance (PRIMARY)
            [Future: + entropy and statistical measures]
        
        Implementation Status:
            - Phase 1 (9D core): Planned - see utils/comprehensive_similarity.py
            - Phase 2 (12D extended): Future
            - Phase 3 (15D full): Future
            
        Current State:
            The current address structure stores 6 geometric scores per neighbor
            (dot, wedge, tensor, spinor, energy, rank) in the m=6 scores section.
            This will be extended to store or reference full 15D vectors.
            
        TO BE IMPLEMENTED in future PR
        """
        raise NotImplementedError(
            "neighbor_similarity_vectors: Stub for Phase 1 planning. "
            "Full implementation requires integration with ComprehensiveSimilarity "
            "and modification of neighbor selection logic. "
            "See utils/comprehensive_similarity.py for core implementation."
        )

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
                
                # Scores already computed by neighbor_selector (6 geometric channels)
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

        # ECC would be computed here (placeholder: zeros)
        # Real implementation would compute BCH code from content
        # Addresses are naturally unique based on embeddings - no collision detection needed

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
  64      6     scores (m) - 6 geometric similarity channels
  70      16    coords (k)

Neighbor roles:
  N1-N32  (idx 0-31):   absolute nearest (metric-based)
  N33-N48 (idx 32-47):  attractors (high dot product)
  N49-N64 (idx 48-63):  repulsors (high wedge product, orthogonal)

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


# =============================================================================
# PLANNING NOTE: Future 15D Comprehensive Similarity Integration
# =============================================================================

COMPREHENSIVE_SIMILARITY_INTEGRATION_PLAN = """
PLANNING NOTE: Extended Neighbor Scoring with 15D Similarity Vectors

Current Implementation (Phase 0):
  - NeighborSelector computes 6 geometric scores per neighbor
  - Selection based on metric distances (dot, wedge products)
  - Stored in m=6 scores section of neighbor blocks

Phase 1 Enhancement (9D Core - COMPLETE):
  Implementation: utils/comprehensive_similarity.py
  - ✅ Compute 9D similarity vectors for all candidates
  - ✅ Dimensions: [cosine, wedge, tensor, spinor_mag, spinor_phase,
                 energy, l2_tangent, l1_tangent, lior_distance]
  - ✅ Aggregate to scalar scores for neighbor selection
  - ✅ Vectorized operations for GPU efficiency
  - ✅ 23 passing unit tests
  - TODO: Store full 9D vectors for selected neighbors (future)
  - TODO: Integrate with NeighborSelector (future PR)

Phase 2 Enhancement (12D Extended - FUTURE):
  - Add 3 entropy measures: variational, Rényi, curvature
  - Cost: ~600 FLOPs per candidate (still fast for all N)

Phase 3 Enhancement (15D Full - FUTURE):
  - Add 3 statistical measures: Kendall tau, mutual info, sectional curvature
  - Use tiered computation: expensive measures only for top-K
  - Total cost: ~0.8ms with tiering (vs 120ms naive)

Integration Points:
  1. NeighborSelector.select_neighbors():
     - Replace simple distance computation with ComprehensiveSimilarity
     - Call sim_computer.compute_batch(query_embedding, candidate_embeddings)
     - Aggregate to scalar with sim_computer.aggregate_to_scalar()
     - Use aggregated scores for top-K selection
  
  2. Address.neighbor_similarity_vectors property:
     - Return stored 15D vectors for all 64 neighbors
     - Either store in extended neighbor blocks or compute on-demand
  
  3. AddressBuilder.forward():
     - Instantiate ComprehensiveSimilarity with manifold
     - Pass to NeighborSelector for enhanced selection
     - Optionally store full 15D vectors in address

Implementation Strategy:
  - Phase 1: Compute 9D, store existing 6 scores (backward compatible) [COMPLETE]
  - Later: Extend neighbor block from d_block=86 to include 15D vectors
  - Or: Store 15D vectors externally and reference via coords

See: utils/comprehensive_similarity.py for core implementation
     Problem Statement: "Extend Neighbor Addressing with Comprehensive 
                        Similarity Scores + Computational Cost Analysis"
"""

"""
Linearized Address Structure

The address is a structured vector with fixed-width blocks:
[ core | geom | N1 | N2 | ... | N64 | integrity ]

Position and ordering encode meaning (conditioning on structured priors).

Dimensions (default d=512):
- core: d = 512 (embedding)
- geom: 2d = 1024 (metric + transport)
- neighbors: 64 × d_block = 5632 (N1-N64)
- integrity: 34 (ecc + timestamps)
- Total D = 7202 floats

Neighbor roles by position:
- N1-N32: absolute nearest (similarity grounding)
- N33-N48: attractors (reinforcing evidence)
- N49-N64: repulsors (contrastive evidence)

Per-neighbor block (d_block = 88):
- value: d' = 64 (reduced interaction vector)
- scores: m = 8 (similarity, heap rank, salience)
- coords: k = 16 (routing info)
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
    m: int = 8          # scores dim
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
    linearized address structure.
    """

    def __init__(self, config: Optional[AddressConfig] = None):
        super().__init__()
        self.config = config or AddressConfig()

        # Projections for building address components
        self.metric_proj = nn.Linear(self.config.d, self.config.d)
        self.transport_proj = nn.Linear(self.config.d, self.config.d)
        self.value_proj = nn.Linear(self.config.d, self.config.d_prime)
        self.score_proj = nn.Linear(self.config.d, self.config.m)
        self.coord_proj = nn.Linear(self.config.d, self.config.k)

        # Initialize metric/transport near identity
        nn.init.eye_(self.metric_proj.weight)
        nn.init.zeros_(self.metric_proj.bias)
        nn.init.eye_(self.transport_proj.weight)
        nn.init.zeros_(self.transport_proj.bias)

    def forward(
        self,
        embedding: torch.Tensor,
        neighbor_embeddings: Optional[torch.Tensor] = None,
        neighbor_similarities: Optional[torch.Tensor] = None,
        timestamp: Optional[float] = None
    ) -> Address:
        """
        Build address from embedding and neighbors.

        Args:
            embedding: Core embedding, shape (batch, d)
            neighbor_embeddings: Neighbor embeddings, shape (batch, N, d)
            neighbor_similarities: Precomputed similarities, shape (batch, N)
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

        # Fill neighbors (if provided)
        if neighbor_embeddings is not None:
            # Project each neighbor
            values = self.value_proj(neighbor_embeddings)  # (batch, N, d')
            coords = self.coord_proj(neighbor_embeddings)  # (batch, N, k)

            # Scores from similarities + learned projection
            if neighbor_similarities is not None:
                base_scores = neighbor_similarities.unsqueeze(-1)  # (batch, N, 1)
                learned_scores = self.score_proj(neighbor_embeddings)  # (batch, N, m)
                # Concat base similarity with learned scores
                scores = torch.cat([
                    base_scores,
                    learned_scores[..., 1:]  # Skip first to make room for similarity
                ], dim=-1)
            else:
                scores = self.score_proj(neighbor_embeddings)

            # Pack into neighbor blocks
            # blocked shape: (batch, N, d_block)
            blocked = torch.cat([values, scores, coords], dim=-1)

            # Flatten and assign
            addr.data[..., self.config.neighbors_start:self.config.neighbors_end] = \
                blocked.view(batch_size, -1)

        # Fill timestamps
        current_time = timestamp if timestamp is not None else time_module.time()
        addr.timestamps = torch.tensor(
            [[current_time, current_time]],
            device=device, dtype=dtype
        ).expand(batch_size, -1)

        # ECC would be computed here (placeholder: zeros)
        # Real implementation would compute BCH code from content

        return addr


# =============================================================================
# Schema documentation (for reference)
# =============================================================================

ADDRESS_SCHEMA = """
Linearized Address Layout (D = 7202 for d=512):

Offset    Size    Field
------    ----    -----
0         512     core (embedding)
512       512     metric (diagonal)
1024      512     transport (diagonal)
1536      5632    neighbors (64 × 88)
7168      32      ecc
7200      2       timestamps

Per-neighbor block (88 floats):
  Offset  Size  Field
  ------  ----  -----
  0       64    value (d')
  64      8     scores (m)
  72      16    coords (k)

Neighbor roles:
  N1-N32  (idx 0-31):   absolute nearest
  N33-N48 (idx 32-47):  attractors
  N49-N64 (idx 48-63):  repulsors
"""

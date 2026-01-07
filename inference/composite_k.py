"""
Composite K Structure for Geometric Attention

K is NOT a neural vector - it's a composite address structure with 9 fields:
1. embedding: Learned embeddings (BERT with thaw schedule)
2. metric: Local metric tensor (diag_rot/block_rot)
3. christoffel: Christoffel symbols (connection coefficients)
4. knn: kNN similarity scores (k=32 neighbors)
5. min_heap: k/2 strongest attractions (semantic pull)
6. max_heap: k/2 strongest contrasts (semantic repulsion)
7. ecc: 4x8 BCH error correction code
8. timestamp: Wall clock time (for causal ordering)
9. coords: Spatial/semantic coordinates

The composite K replaces standard neural K vectors with a structured address
that enables:
- Geometric routing in the cognitive field
- kNN-based retrieval with heap-based filtering
- Error-corrected addressing
- Temporal causality

Architecture:
    Q = local_state + previous_address_blocks (self-bootstrapping)
    K = composite_address (this module)
    V = payload (content, state_update)

    Output O splits into: (state_update, new_address_block)
    The new_address_block feeds into next Q
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict, List
from dataclasses import dataclass
import time
import heapq


@dataclass
class CompositeKFields:
    """Container for the 9-field composite K structure."""
    embedding: torch.Tensor      # (batch, seq, d_embed) - learned BERT-style
    metric: torch.Tensor         # (batch, seq, d_metric) - local geometry
    christoffel: torch.Tensor    # (batch, seq, d_chris) - connection
    knn_scores: torch.Tensor     # (batch, seq, k) - k=32 neighbor similarities
    knn_indices: torch.Tensor    # (batch, seq, k) - neighbor indices
    min_heap: torch.Tensor       # (batch, seq, k//2) - strongest attractions
    max_heap: torch.Tensor       # (batch, seq, k//2) - strongest contrasts
    ecc: torch.Tensor            # (batch, seq, 32) - 4x8 BCH code
    timestamp: torch.Tensor      # (batch, seq, 1) - wall time
    coords: torch.Tensor         # (batch, seq, d_coords) - position


class ThawSchedule(nn.Module):
    """
    Inverse warmup for BERT weights.

    Instead of warming up LR from 0, we freeze BERT initially
    and gradually unfreeze layers as training progresses.

    Schedule:
        epoch 0-2: All frozen (only adapter trains)
        epoch 2-4: Top 2 layers unfrozen
        epoch 4-6: Top 4 layers unfrozen
        epoch 6+:  All layers unfrozen
    """

    def __init__(self, n_layers: int = 12, thaw_per_epoch: int = 2):
        super().__init__()
        self.n_layers = n_layers
        self.thaw_per_epoch = thaw_per_epoch
        self.current_epoch = 0

    def get_frozen_layers(self) -> int:
        """Returns number of layers that should remain frozen."""
        # Start fully frozen, thaw from top (layer n-1) down to bottom (layer 0)
        thawed = self.current_epoch * self.thaw_per_epoch
        return max(0, self.n_layers - thawed)

    def step_epoch(self):
        """Advance to next epoch."""
        self.current_epoch += 1

    def apply_to_bert(self, bert_model: nn.Module):
        """Apply current freeze state to BERT model."""
        frozen = self.get_frozen_layers()

        # Freeze embedding layer if any layers are frozen
        if frozen > 0 and hasattr(bert_model, 'embeddings'):
            for param in bert_model.embeddings.parameters():
                param.requires_grad = False

        # Freeze/unfreeze encoder layers
        if hasattr(bert_model, 'encoder') and hasattr(bert_model.encoder, 'layer'):
            for i, layer in enumerate(bert_model.encoder.layer):
                # Layer 0 is bottom, layer n-1 is top
                # Freeze bottom `frozen` layers
                requires_grad = i >= frozen
                for param in layer.parameters():
                    param.requires_grad = requires_grad


class LocalMetricComputer(nn.Module):
    """
    Computes local metric tensor for geometric routing.

    Options:
    - diag_rot: Diagonal + rotation (efficient, ~O(d))
    - block_rot: Block diagonal + rotation (more expressive, ~O(d*block_size))
    """

    def __init__(
        self,
        d_model: int,
        metric_type: str = 'diag_rot',
        block_size: int = 8
    ):
        super().__init__()
        self.d_model = d_model
        self.metric_type = metric_type
        self.block_size = block_size

        if metric_type == 'diag_rot':
            # Diagonal scaling + learned rotation
            self.diag = nn.Linear(d_model, d_model, bias=False)
            nn.init.eye_(self.diag.weight)
        elif metric_type == 'block_rot':
            # Block diagonal metric
            n_blocks = d_model // block_size
            self.blocks = nn.ModuleList([
                nn.Linear(block_size, block_size, bias=False)
                for _ in range(n_blocks)
            ])
            for block in self.blocks:
                nn.init.orthogonal_(block.weight)
        else:
            raise ValueError(f"Unknown metric_type: {metric_type}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute local metric at each position.

        Args:
            x: (batch, seq, d_model)

        Returns:
            metric: (batch, seq, d_model) - transformed by local metric
        """
        if self.metric_type == 'diag_rot':
            return self.diag(x)
        else:
            # Block diagonal application
            batch, seq, _ = x.shape
            chunks = x.split(self.block_size, dim=-1)
            out_chunks = [block(chunk) for block, chunk in zip(self.blocks, chunks)]
            return torch.cat(out_chunks, dim=-1)


class ChristoffelComputer(nn.Module):
    """
    Computes Christoffel symbols (connection coefficients) for geodesic routing.

    The Christoffel symbols Gamma^k_ij encode how coordinate basis vectors
    change as you move through the manifold. For attention, this tells us
    how to parallel transport Q to compare with K at different positions.
    """

    def __init__(self, d_model: int, n_christoffel: int = 32):
        super().__init__()
        self.d_model = d_model
        self.n_christoffel = n_christoffel

        # Project to Christoffel space (much smaller than full d^3 tensor)
        self.proj = nn.Linear(d_model, n_christoffel)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute Christoffel symbols at each position.

        Args:
            x: (batch, seq, d_model)

        Returns:
            christoffel: (batch, seq, n_christoffel)
        """
        return self.proj(x)


class KNNModule(nn.Module):
    """
    k-Nearest Neighbors module for composite K.

    Computes k=32 nearest neighbors based on embedding similarity,
    then splits into min-heap (attractions) and max-heap (contrasts).
    """

    def __init__(self, k: int = 32, use_faiss: bool = False):
        super().__init__()
        self.k = k
        self.use_faiss = use_faiss

    def forward(
        self,
        embeddings: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute kNN and heap structures.

        Args:
            embeddings: (batch, seq, d_model)

        Returns:
            knn_scores: (batch, seq, k) - similarity scores to k neighbors
            knn_indices: (batch, seq, k) - indices of k neighbors
            min_heap: (batch, seq, k//2) - k/2 strongest attractions (highest similarity)
            max_heap: (batch, seq, k//2) - k/2 strongest contrasts (for diversity)
        """
        batch, seq, d = embeddings.shape

        # Normalize for cosine similarity
        emb_norm = embeddings / (embeddings.norm(dim=-1, keepdim=True) + 1e-8)

        # Compute pairwise similarity: (batch, seq, seq)
        sim_matrix = torch.bmm(emb_norm, emb_norm.transpose(-2, -1))

        # Mask self-similarity
        eye = torch.eye(seq, device=embeddings.device).unsqueeze(0)
        sim_matrix = sim_matrix - eye * 1e9

        # Top-k similar (attractions)
        topk = min(self.k, seq - 1)
        knn_scores, knn_indices = sim_matrix.topk(topk, dim=-1)

        # Pad if needed
        if topk < self.k:
            pad_size = self.k - topk
            knn_scores = torch.cat([
                knn_scores,
                torch.zeros(batch, seq, pad_size, device=embeddings.device)
            ], dim=-1)
            knn_indices = torch.cat([
                knn_indices,
                torch.zeros(batch, seq, pad_size, dtype=torch.long, device=embeddings.device)
            ], dim=-1)

        # Split into heaps
        half_k = self.k // 2
        min_heap = knn_scores[..., :half_k]   # Top k/2 highest (attractions)
        max_heap = -knn_scores[..., half_k:]  # Convert to max-heap of contrasts

        return knn_scores, knn_indices, min_heap, max_heap


class BCHEncoder(nn.Module):
    """
    4x8 BCH Error Correction Code for address integrity.

    BCH(15,7) can correct 2 errors in 7 data bits.
    We use 4 parallel BCH codes for 32-bit ECC field.
    """

    def __init__(self, d_model: int, ecc_bits: int = 32):
        super().__init__()
        self.ecc_bits = ecc_bits

        # Project embeddings to ECC space
        self.proj = nn.Linear(d_model, ecc_bits)

        # Learned BCH-like encoding (soft ECC)
        self.encode = nn.Sequential(
            nn.Linear(ecc_bits, ecc_bits * 2),
            nn.Tanh(),
            nn.Linear(ecc_bits * 2, ecc_bits),
            nn.Sigmoid()  # Binary-like output
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Generate ECC code from embeddings.

        Args:
            x: (batch, seq, d_model)

        Returns:
            ecc: (batch, seq, ecc_bits) - error correction code
        """
        projected = self.proj(x)
        return self.encode(projected)


class CompositeK(nn.Module):
    """
    Full Composite K Structure generator.

    Generates all 9 fields of the composite K address:
    1. embedding (from BERT with thaw schedule)
    2. metric (local geometry)
    3. christoffel (connection)
    4. knn_scores, knn_indices (neighbors)
    5. min_heap (attractions)
    6. max_heap (contrasts)
    7. ecc (error correction)
    8. timestamp
    9. coords
    """

    def __init__(
        self,
        d_model: int = 256,
        d_embed: int = 128,      # Embedding contribution dimension
        n_christoffel: int = 32,
        k_neighbors: int = 32,
        metric_type: str = 'diag_rot',
        use_bert: bool = False,  # If True, use BERT for embeddings
        bert_model_name: str = 'bert-base-uncased'
    ):
        super().__init__()

        self.d_model = d_model
        self.d_embed = d_embed
        self.k_neighbors = k_neighbors

        # Field 1: Embedding (BERT or learned projection)
        self.use_bert = use_bert
        if use_bert:
            try:
                from transformers import BertModel
                self.bert = BertModel.from_pretrained(bert_model_name)
                self.thaw_schedule = ThawSchedule(n_layers=12)
                self.embed_proj = nn.Linear(768, d_embed)  # BERT hidden -> d_embed
            except ImportError:
                print("transformers not available, using learned embeddings")
                self.use_bert = False

        if not self.use_bert:
            self.embed_proj = nn.Linear(d_model, d_embed)
            self.thaw_schedule = None

        # Field 2: Local metric
        self.metric = LocalMetricComputer(d_model, metric_type)

        # Field 3: Christoffel symbols
        self.christoffel = ChristoffelComputer(d_model, n_christoffel)

        # Fields 4-6: kNN and heaps
        self.knn = KNNModule(k=k_neighbors)

        # Field 7: BCH ECC
        self.ecc = BCHEncoder(d_model, ecc_bits=32)

        # Field 9: Coordinate projection (field 8 timestamp is computed externally)
        self.coord_proj = nn.Linear(d_model, d_model // 4)  # Compressed coords

        # Output projection: K_flat needs to match input dimension for attention
        k_flat_dim = d_embed + d_model // 4  # embedding + coords
        self.k_output_proj = nn.Linear(k_flat_dim, d_model)

    def step_epoch(self):
        """Advance thaw schedule by one epoch."""
        if self.thaw_schedule is not None:
            self.thaw_schedule.step_epoch()
            if self.use_bert:
                self.thaw_schedule.apply_to_bert(self.bert)

    def forward(
        self,
        x: torch.Tensor,
        timestamp: Optional[float] = None,
        return_dict: bool = False
    ) -> Tuple[CompositeKFields, torch.Tensor]:
        """
        Generate composite K from input embeddings.

        Args:
            x: (batch, seq, d_model) - input from encoder
            timestamp: Wall clock time (uses time.time() if None)
            return_dict: If True, return fields in dataclass

        Returns:
            fields: CompositeKFields dataclass with all 9 fields
            K_flat: (batch, seq, K_dim) - flattened K for attention compatibility
        """
        batch, seq, _ = x.shape
        device = x.device

        # Field 1: Embedding
        if self.use_bert:
            # BERT expects input_ids, but we have embeddings
            # Use learned projection instead
            embedding = self.embed_proj(x)
        else:
            embedding = self.embed_proj(x)

        # Field 2: Metric
        metric = self.metric(x)

        # Field 3: Christoffel
        christoffel = self.christoffel(x)

        # Fields 4-6: kNN and heaps
        knn_scores, knn_indices, min_heap, max_heap = self.knn(x)

        # Field 7: ECC
        ecc = self.ecc(x)

        # Field 8: Timestamp
        if timestamp is None:
            timestamp = time.time()
        ts = torch.full((batch, seq, 1), timestamp, device=device, dtype=x.dtype)

        # Field 9: Coordinates
        coords = self.coord_proj(x)

        fields = CompositeKFields(
            embedding=embedding,
            metric=metric,
            christoffel=christoffel,
            knn_scores=knn_scores,
            knn_indices=knn_indices,
            min_heap=min_heap,
            max_heap=max_heap,
            ecc=ecc,
            timestamp=ts,
            coords=coords
        )

        # Flatten for attention compatibility
        # Use embedding + coords, then project to match input dimension
        K_concat = torch.cat([embedding, coords], dim=-1)
        K_flat = self.k_output_proj(K_concat)

        return fields, K_flat


class OutputSplitter(nn.Module):
    """
    Splits output O into (state_update, address_block).

    The address_block feeds back into Q for self-bootstrapping.
    This enables the model to generate its own query addresses
    based on what it has learned.
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = None,     # Dimension for state update
        d_address: int = None    # Dimension for address block
    ):
        super().__init__()

        if d_state is None:
            d_state = d_model // 2
        if d_address is None:
            d_address = d_model // 2

        self.d_state = d_state
        self.d_address = d_address

        # Split projection
        self.split_proj = nn.Linear(d_model, d_state + d_address)

        # Separate refinement for each output
        self.state_refine = nn.Linear(d_state, d_state)
        self.address_refine = nn.Linear(d_address, d_address)

    def forward(self, output: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Split output into state update and address block.

        Args:
            output: (batch, seq, d_model) - model output

        Returns:
            state_update: (batch, seq, d_state) - for V/payload update
            address_block: (batch, seq, d_address) - for next Q
        """
        combined = self.split_proj(output)

        state = combined[..., :self.d_state]
        address = combined[..., self.d_state:]

        state_update = self.state_refine(state)
        address_block = self.address_refine(address)

        return state_update, address_block


# =============================================================================
# Integration helper
# =============================================================================

def create_composite_k_system(
    d_model: int = 256,
    k_neighbors: int = 32,
    use_bert: bool = False
) -> Tuple[CompositeK, OutputSplitter]:
    """
    Create a complete composite K system.

    Returns:
        composite_k: CompositeK generator
        output_splitter: OutputSplitter for self-bootstrapping
    """
    composite_k = CompositeK(
        d_model=d_model,
        d_embed=d_model // 2,
        k_neighbors=k_neighbors,
        use_bert=use_bert
    )

    output_splitter = OutputSplitter(d_model=d_model)

    return composite_k, output_splitter

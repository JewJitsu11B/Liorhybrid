"""
Multimodal Embeddings

Converts different modalities into unified d_model dimensional space.

Modalities:
- Text: Learned token embeddings + RoPE (no length limit)
- Images: Patch embeddings (ViT-style)
- Video: Frame sampling + temporal encoding

All modalities project to same d_model space for geometric attention.

RoPE (Rotary Position Embedding):
- No fixed sequence length limit - extrapolates to any length
- Applied in attention, not added to embeddings
- Better length generalization than learned positional embeddings
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List, Union
import math


class RotaryPositionEmbedding(nn.Module):
    """
    Rotary Position Embedding (RoPE) - NO LENGTH LIMIT.

    Instead of adding positional embeddings, RoPE rotates query/key vectors
    by position-dependent angles. This allows extrapolation to any sequence length.

    Reference: "RoFormer: Enhanced Transformer with Rotary Position Embedding"

    Properties:
    - No max_seq_len limit - works for any sequence length
    - Relative position aware (q_m * k_n depends on m-n)
    - Compatible with linear attention
    """

    def __init__(self, dim: int, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.base = base
        # Precompute inverse frequencies
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq, persistent=False)
        self._seq_len_cached = 0
        self._cos_cached = None
        self._sin_cached = None

    def _update_cos_sin_cache(self, seq_len: int, device: torch.device, dtype: torch.dtype):
        """Update cached cos/sin values if sequence length changed."""
        if seq_len > self._seq_len_cached:
            self._seq_len_cached = seq_len
            t = torch.arange(seq_len, device=device, dtype=dtype)
            freqs = torch.einsum('i,j->ij', t, self.inv_freq.to(device=device, dtype=dtype))
            # Duplicate for pairing: [f0, f0, f1, f1, ...]
            emb = torch.cat([freqs, freqs], dim=-1)
            self._cos_cached = emb.cos()[None, None, :, :]  # (1, 1, seq, dim)
            self._sin_cached = emb.sin()[None, None, :, :]

    def forward(self, x: torch.Tensor, seq_len: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get cos/sin for rotary embedding.

        Args:
            x: Input tensor to get device/dtype from
            seq_len: Sequence length (if None, infer from x)

        Returns:
            (cos, sin): Each (1, 1, seq_len, dim)
        """
        if seq_len is None:
            seq_len = x.shape[-2]
        self._update_cos_sin_cache(seq_len, x.device, x.dtype)
        return (
            self._cos_cached[:, :, :seq_len, :],
            self._sin_cached[:, :, :seq_len, :]
        )


def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary position embedding to query and key tensors.

    Args:
        q: Query tensor (batch, heads, seq, dim)
        k: Key tensor (batch, heads, seq, dim)
        cos: Cosine values (1, 1, seq, dim)
        sin: Sine values (1, 1, seq, dim)

    Returns:
        (q_rotated, k_rotated): Rotated tensors
    """
    def rotate_half(x):
        """Rotate half the hidden dims."""
        x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
        return torch.cat([-x2, x1], dim=-1)

    q_rotated = (q * cos) + (rotate_half(q) * sin)
    k_rotated = (k * cos) + (rotate_half(k) * sin)
    return q_rotated, k_rotated


class TextEmbedding(nn.Module):
    """
    Text token embeddings with RoPE (Rotary Position Embedding).

    NO SEQUENCE LENGTH LIMIT - RoPE extrapolates to any length.

    - Token embedding lookup (learned)
    - RoPE for positions (applied in attention, stored here for convenience)
    - Dropout
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        max_seq_len: int = 2048,  # Ignored - kept for API compatibility
        dropout: float = 0.1,
        rope_base: float = 10000.0
    ):
        super().__init__()

        self.d_model = d_model

        # Token embeddings (learned)
        self.token_embedding = nn.Embedding(vocab_size, d_model)

        # RoPE - NO LENGTH LIMIT
        self.rotary_emb = RotaryPositionEmbedding(d_model, base=rope_base)

        self.dropout = nn.Dropout(dropout)

        # Initialize embeddings
        nn.init.normal_(self.token_embedding.weight, std=0.02)

    def forward(
        self,
        token_ids: torch.Tensor,
        spans: Optional[List[List[Tuple[int, int]]]] = None,
        return_span_emb: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        Embed text tokens, optionally with span pooling.

        Args:
            token_ids: (batch, seq_len) integer token IDs - ANY LENGTH
            spans: Optional list of span boundaries per batch item.
                   Each item is List[(start_token_idx, end_token_idx)]
            return_span_emb: If True and spans provided, also return span embeddings

        Returns:
            If return_span_emb=False or spans=None:
                embeddings: (batch, seq_len, d_model)
            If return_span_emb=True and spans provided:
                Tuple of (token_embeddings, span_embeddings, span_mask)
                - token_embeddings: (batch, seq_len, d_model)
                - span_embeddings: (batch, max_spans, d_model) - pooled from token embeddings
                - span_mask: (batch, max_spans) - True for valid spans
        """
        # Token embeddings only - RoPE applied in attention
        token_emb = self.token_embedding(token_ids)  # (batch, seq_len, d_model)
        token_emb = self.dropout(token_emb)

        if not return_span_emb or spans is None:
            return token_emb

        # Compute span embeddings by pooling token embeddings within each span
        batch_size = token_ids.shape[0]
        device = token_ids.device

        # Find max spans for padding
        max_spans = max(len(s) for s in spans) if spans else 1

        span_emb = torch.zeros(batch_size, max_spans, self.d_model, device=device, dtype=token_emb.dtype)
        span_mask = torch.zeros(batch_size, max_spans, dtype=torch.bool, device=device)

        for b, batch_spans in enumerate(spans):
            for s, (start, end) in enumerate(batch_spans):
                if start >= end or end > token_emb.shape[1]:
                    continue

                # Mean pooling over tokens in span
                span_tokens = token_emb[b, start:end]  # (span_len, d_model)
                span_emb[b, s] = span_tokens.mean(dim=0)
                span_mask[b, s] = True

        return token_emb, span_emb, span_mask

    def get_rotary_emb(self, x: torch.Tensor, seq_len: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get RoPE cos/sin for attention layers."""
        return self.rotary_emb(x, seq_len)


class ImagePatchEmbedding(nn.Module):
    """
    Image patch embeddings (ViT-style).

    Divides image into patches, projects each patch to d_model.

    For image of size (C, H, W):
    - Divide into (H/patch_size) × (W/patch_size) patches
    - Each patch is (C, patch_size, patch_size)
    - Project to d_model with conv2d

    Reference: "An Image is Worth 16x16 Words" (ViT paper)
    """

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        d_model: int = 512,
        dropout: float = 0.1
    ):
        super().__init__()

        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2

        # Patch projection (convolution with stride = patch_size)
        self.patch_proj = nn.Conv2d(
            in_channels,
            d_model,
            kernel_size=patch_size,
            stride=patch_size
        )

        # Class token (optional, for classification tasks)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))

        # Position embeddings
        self.position_embedding = nn.Parameter(
            torch.zeros(1, self.n_patches + 1, d_model)  # +1 for cls token
        )

        self.dropout = nn.Dropout(dropout)

        # Initialize
        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.position_embedding, std=0.02)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Embed image patches.

        Args:
            images: (batch, C, H, W) images

        Returns:
            embeddings: (batch, n_patches + 1, d_model)
        """
        batch_size = images.shape[0]
        device = images.device

        # Project patches: (batch, C, H, W) -> (batch, d_model, H/p, W/p)
        x = self.patch_proj(images)

        # Flatten patches: (batch, d_model, H/p, W/p) -> (batch, d_model, n_patches)
        x = x.flatten(2)

        # Transpose: (batch, d_model, n_patches) -> (batch, n_patches, d_model)
        x = x.transpose(1, 2)

        # Add class token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # (batch, 1, d_model)
        x = torch.cat([cls_tokens, x], dim=1)  # (batch, n_patches + 1, d_model)

        # Add positional embeddings
        x = x + self.position_embedding

        x = self.dropout(x)

        return x


class VideoEmbedding(nn.Module):
    """
    Video embeddings via frame sampling + temporal encoding.

    Architecture:
    1. Sample N frames uniformly from video
    2. Embed each frame with ImagePatchEmbedding
    3. Add temporal positional encoding
    4. Optionally pool across time

    For video of shape (T, C, H, W):
    - Sample frames at indices [0, T/N, 2T/N, ..., (N-1)T/N]
    - Embed each frame -> (N, n_patches, d_model)
    - Add temporal encoding
    """

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        d_model: int = 512,
        n_frames: int = 8,
        dropout: float = 0.1
    ):
        super().__init__()

        self.n_frames = n_frames

        # Frame embedding (reuse image patch embedding)
        self.frame_embedding = ImagePatchEmbedding(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=in_channels,
            d_model=d_model,
            dropout=dropout
        )

        # Temporal position embeddings
        n_patches = self.frame_embedding.n_patches
        self.temporal_embedding = nn.Parameter(
            torch.zeros(1, n_frames, n_patches + 1, d_model)
        )

        nn.init.normal_(self.temporal_embedding, std=0.02)

    def forward(
        self,
        video: torch.Tensor,
        return_sequence: bool = True
    ) -> torch.Tensor:
        """
        Embed video frames.

        Args:
            video: (batch, T, C, H, W) video tensor
            return_sequence: If True, return all frame tokens.
                            If False, pool to single representation.

        Returns:
            If return_sequence:
                embeddings: (batch, n_frames * (n_patches + 1), d_model)
            Else:
                embeddings: (batch, 1, d_model)
        """
        batch_size, T, C, H, W = video.shape
        device = video.device

        # Sample frames uniformly
        frame_indices = torch.linspace(0, T - 1, self.n_frames, device=device).long()
        sampled_frames = video[:, frame_indices]  # (batch, n_frames, C, H, W)

        # Embed each frame
        frame_embeddings = []
        for i in range(self.n_frames):
            frame = sampled_frames[:, i]  # (batch, C, H, W)
            emb = self.frame_embedding(frame)  # (batch, n_patches + 1, d_model)
            frame_embeddings.append(emb)

        # Stack frames: (batch, n_frames, n_patches + 1, d_model)
        frame_embeddings = torch.stack(frame_embeddings, dim=1)

        # Add temporal embeddings
        frame_embeddings = frame_embeddings + self.temporal_embedding

        if return_sequence:
            # Flatten to sequence: (batch, n_frames * (n_patches + 1), d_model)
            embeddings = frame_embeddings.flatten(1, 2)
        else:
            # Pool across frames (mean over time and spatial tokens)
            embeddings = frame_embeddings.mean(dim=(1, 2), keepdim=True)  # (batch, 1, d_model)

        return embeddings


class MultimodalEmbedding(nn.Module):
    """
    Unified multimodal embedding layer.

    Handles text, images, and video in a single interface.

    Usage:
        embedder = MultimodalEmbedding(vocab_size=32000, d_model=512)

        # Text
        text_emb = embedder(text_tokens, modality='text')

        # Image
        img_emb = embedder(images, modality='image')

        # Video
        vid_emb = embedder(video_frames, modality='video')
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        max_seq_len: int = 2048,
        img_size: int = 224,
        patch_size: int = 16,
        n_video_frames: int = 8,
        dropout: float = 0.1
    ):
        super().__init__()

        self.d_model = d_model

        # Text embeddings
        self.text_embedding = TextEmbedding(
            vocab_size=vocab_size,
            d_model=d_model,
            max_seq_len=max_seq_len,
            dropout=dropout
        )

        # Image embeddings
        self.image_embedding = ImagePatchEmbedding(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=3,
            d_model=d_model,
            dropout=dropout
        )

        # Video embeddings
        self.video_embedding = VideoEmbedding(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=3,
            d_model=d_model,
            n_frames=n_video_frames,
            dropout=dropout
        )

        # Modality type embeddings (to distinguish modalities)
        self.modality_embedding = nn.Embedding(3, d_model)  # [text, image, video]

    def forward(
        self,
        x: torch.Tensor,
        modality: str = 'text'
    ) -> torch.Tensor:
        """
        Embed input from specified modality.

        Args:
            x: Input tensor (shape depends on modality)
               - text: (batch, seq_len) token IDs
               - image: (batch, C, H, W) images
               - video: (batch, T, C, H, W) video frames
            modality: One of ['text', 'image', 'video']

        Returns:
            embeddings: (batch, seq_len, d_model)
        """
        if modality == 'text':
            emb = self.text_embedding(x)
            modality_id = 0
        elif modality == 'image':
            emb = self.image_embedding(x)
            modality_id = 1
        elif modality == 'video':
            emb = self.video_embedding(x, return_sequence=True)
            modality_id = 2
        else:
            raise ValueError(f"Unknown modality: {modality}")

        # Add modality type embedding
        modality_emb = self.modality_embedding(
            torch.tensor([modality_id], device=x.device)
        )
        emb = emb + modality_emb

        return emb


# =============================================================================
# MULTI-DOMAIN COARSE/FINE EMBEDDING SYSTEM (SCAFFOLD)
# =============================================================================
#
# Architecture Overview:
# ---------------------
# K (key) in attention is factorized into domain slots, each with coarse + fine granularity:
#
#   K = [coding_coarse | coding_fine | math_coarse | math_fine | legal_coarse | legal_fine | ...]
#
# Each domain captures different semantic spaces with different granularity levels:
#
#   Domain        Fine (syntax/rules)           Coarse (patterns/concepts)
#   ------        ------------------            -------------------------
#   coding        syntax rules, operators       architectural patterns, idioms
#   math          formulas, proofs              conceptual frameworks, intuition
#   legal         clauses, statutes             principles, precedents
#   ethics        specific rules                philosophical frameworks
#
# Attention Head Structure (Bifaceted):
# ------------------------------------
# Each attention head has two "faces":
#   - Granularity face: attends across fine <-> coarse within same domain
#   - Domain face: attends across domains at same granularity level
#
# This allows learning both:
#   - "code syntax is like math formulas" (cross-domain, same granularity)
#   - "this code pattern relates to this syntax rule" (same domain, cross-granularity)
#
# Parameter Layout:
# ----------------
# config['embedding_domains'] = ['coding', 'math', 'legal', 'ethics']  # n_domains = 4
# config['embedding_dim_per_slot'] = 64                                 # per coarse/fine slot
# config['bifaceted_heads'] = True
#
# Total K dimension = n_domains × 2 × dim_per_slot = 4 × 2 × 64 = 512
#
# Implementation Steps:
# --------------------
# 1. DomainTokenizer: Per-domain tokenizers with coarse/fine granularity
#    - Fine: BPE/SentencePiece with small vocab (captures syntax)
#    - Coarse: BPE/SentencePiece with large vocab (captures concepts)
#
# 2. DomainEmbedding(nn.Module):
#    - Embedding tables per domain × granularity
#    - forward() returns [batch, seq, n_domains × 2 × dim_per_slot]
#
# 3. BifacetedAttention(nn.Module):
#    - Modified Q/K/V projections aware of domain structure
#    - Attention patterns that can attend across granularity OR domain axes
#    - Head specialization: some heads are "granularity heads", others are "domain heads"
#
# 4. Position Encoding:
#    - RoPE applied per-domain (positions relative within each domain's token stream)
#    - Cross-domain position encoding for alignment
#
# Example Usage (when implemented):
# --------------------------------
# class DomainEmbedding(nn.Module):
#     def __init__(self, domains: List[str], dim_per_slot: int):
#         super().__init__()
#         self.domains = domains
#         self.dim_per_slot = dim_per_slot
#
#         # Embedding tables: [domain][granularity]
#         self.embeddings = nn.ModuleDict({
#             f"{domain}_{gran}": nn.Embedding(vocab_size, dim_per_slot)
#             for domain in domains
#             for gran in ['coarse', 'fine']
#         })
#
#     def forward(self, tokens: Dict[str, Dict[str, torch.Tensor]]) -> torch.Tensor:
#         # tokens['coding']['fine'] = (batch, seq_coding_fine)
#         # tokens['math']['coarse'] = (batch, seq_math_coarse)
#         # ... etc
#         # Returns: (batch, total_seq, n_domains * 2 * dim_per_slot)
#         pass
#
# =============================================================================

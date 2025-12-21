"""
Multimodal Embeddings

Converts different modalities into unified d_model dimensional space.

Modalities:
- Text: Learned token embeddings
- Images: Patch embeddings (ViT-style)
- Video: Frame sampling + temporal encoding

All modalities project to same d_model space for geometric attention.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math


class TextEmbedding(nn.Module):
    """
    Learned text token embeddings with positional encoding.

    Standard transformer-style embeddings:
    - Token embedding lookup
    - Learned positional embeddings
    - Dropout
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        max_seq_len: int = 2048,
        dropout: float = 0.1
    ):
        super().__init__()

        self.d_model = d_model

        # Token embeddings
        self.token_embedding = nn.Embedding(vocab_size, d_model)

        # Positional embeddings (learned)
        self.position_embedding = nn.Embedding(max_seq_len, d_model)

        self.dropout = nn.Dropout(dropout)

        # Initialize embeddings
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.position_embedding.weight, std=0.02)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Embed text tokens.

        Args:
            token_ids: (batch, seq_len) integer token IDs

        Returns:
            embeddings: (batch, seq_len, d_model)
        """
        batch_size, seq_len = token_ids.shape
        device = token_ids.device

        # Token embeddings
        token_emb = self.token_embedding(token_ids)  # (batch, seq_len, d_model)

        # Position embeddings
        positions = torch.arange(seq_len, device=device).unsqueeze(0)  # (1, seq_len)
        pos_emb = self.position_embedding(positions)  # (1, seq_len, d_model)

        # Combine
        embeddings = token_emb + pos_emb
        embeddings = self.dropout(embeddings)

        return embeddings


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

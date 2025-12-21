"""
DPR (Dense Passage Retrieval) Integration

Uses pre-trained DPR encoders to generate statistically optimal K/V vectors
for the geometric attention mechanism.

DPR provides:
- Context encoder: Generates Keys and Values from cognitive field state
- Question encoder: Generates Queries from input text

DPR-optimized vectors are maximally distinct and aligned for retrieval
through contrastive training, providing high-quality semantic representations.

References:
- DPR paper: "Dense Passage Retrieval for Open-Domain Question Answering"
- HuggingFace: facebook/dpr-ctx_encoder-single-nq-base
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional

# Import transformers (HuggingFace) - REQUIRED
from transformers import DPRContextEncoder, DPRQuestionEncoder, DPRContextEncoderTokenizer, DPRQuestionEncoderTokenizer


class DPRKeyValueGenerator(nn.Module):
    """
    DPR-based Key/Value generator for geometric attention.

    Uses pre-trained DPR encoders (frozen initially) to generate
    statistically optimal embeddings from cognitive field state.

    Architecture:
        Field state T_ij → Context encoder → K, V vectors
        Input text → Question encoder → Q vectors

    The DPR encoders are frozen to preserve their pre-trained
    statistical quality. Only downstream projection layers are trainable.
    """

    def __init__(
        self,
        d_model: int = 512,
        freeze_encoders: bool = True,
        use_pretrained: bool = True
    ):
        """
        Initialize DPR-based K/V generator.

        Args:
            d_model: Output dimension (projected from DPR's 768)
            freeze_encoders: If True, freeze DPR encoder weights
            use_pretrained: If True, load pre-trained DPR models
        """
        super().__init__()

        self.d_model = d_model
        self.dpr_dim = 768  # DPR base model output dimension

        if use_pretrained:
            # Load pre-trained DPR encoders
            print("Loading pre-trained DPR encoders...")
            self.context_encoder = DPRContextEncoder.from_pretrained(
                'facebook/dpr-ctx_encoder-single-nq-base'
            )
            self.question_encoder = DPRQuestionEncoder.from_pretrained(
                'facebook/dpr-question_encoder-single-nq-base'
            )

            # Load tokenizers
            self.ctx_tokenizer = DPRContextEncoderTokenizer.from_pretrained(
                'facebook/dpr-ctx_encoder-single-nq-base'
            )
            self.q_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained(
                'facebook/dpr-question_encoder-single-nq-base'
            )

            # Freeze encoders if requested
            if freeze_encoders:
                print("Freezing DPR encoder weights...")
                for param in self.context_encoder.parameters():
                    param.requires_grad = False
                for param in self.question_encoder.parameters():
                    param.requires_grad = False

            # Projection layers from DPR's 768-dim to model's d_model
            self.kv_projection = nn.Linear(self.dpr_dim, d_model)
            self.q_projection = nn.Linear(self.dpr_dim, d_model)

            # Initialize projections
            nn.init.xavier_uniform_(self.kv_projection.weight)
            nn.init.xavier_uniform_(self.q_projection.weight)
        else:
            # No DPR: use field directly as structured input (fallback)
            # Field direct projection: will be set dynamically based on field shape
            self.context_encoder = None
            self.question_encoder = None
            self.kv_projection = None
            self.q_projection = None

    @torch.compiler.disable  # Prevent torch.compile from tracing this method
    def _field_to_text(self, field_state: torch.Tensor) -> str:
        """
        Convert field state T_ij to text representation for DPR context encoder.

        This is a simplified approach. In production, you might:
        - Store associated text snippets with field states
        - Use a learned decoder to generate text descriptions
        - Maintain a memory buffer of recent inputs

        For now, we create a simple numeric summary.

        Note: This method is excluded from torch.compile because it uses
        .item() calls which cause graph breaks.

        Args:
            field_state: Cognitive tensor field (N_x, N_y, D, D)

        Returns:
            Text representation of field state
        """
        # Extract field statistics - detach to avoid grad issues
        field_abs = torch.abs(field_state.detach()) if field_state.is_complex() else field_state.detach()

        field_mean = field_abs.mean().item()
        field_std = field_abs.std().item()
        field_max = field_abs.max().item()

        # Create text summary (this is a placeholder)
        text = f"Cognitive field state: mean={field_mean:.3f}, std={field_std:.3f}, max={field_max:.3f}"

        return text

    def generate_kv(
        self,
        field_state: torch.Tensor,
        batch_size: int = 1
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate Keys and Values from cognitive field state using DPR context encoder.

        Args:
            field_state: Cognitive tensor field (N_x, N_y, D, D)
            batch_size: Batch size to match queries

        Returns:
            K: Key vectors (batch, n_tokens, d_model)
            V: Value vectors (batch, n_tokens, d_model)
        """
        if self.context_encoder is not None:
            # Convert field to text representation
            context_text = self._field_to_text(field_state)

            # Tokenize
            inputs = self.ctx_tokenizer(
                context_text,
                return_tensors='pt',
                padding=True,
                truncation=True,
                max_length=512
            )
            # Run DPR encoder on its own device (avoids CPU↔CUDA mismatch during compile/training)
            device = next(self.context_encoder.parameters()).device
            inputs = inputs.to(device)

            # Encode with DPR
            with torch.no_grad() if not self.context_encoder.training else torch.enable_grad():
                outputs = self.context_encoder(**inputs)
                embeddings = outputs.pooler_output  # (1, 768)
        else:
            # Use field tensor directly - preserve geometric structure
            # Field: (N_x, N_y, D, D) complex tensor
            # Reshape to (N_x*N_y, D*D) to preserve spatial structure
            N_x, N_y, D, D_ = field_state.shape

            # Reshape: (N_x, N_y, D, D) → (N_x*N_y, D*D)
            field_reshaped = field_state.reshape(N_x * N_y, D * D_)

            # Convert complex to real if needed (preserve both magnitude and phase info)
            if field_reshaped.is_complex():
                # Stack real and imaginary parts: (N_x*N_y, D*D*2)
                field_real = torch.stack([field_reshaped.real, field_reshaped.imag], dim=-1)
                field_flat = field_real.reshape(N_x * N_y, D * D_ * 2)
            else:
                field_flat = field_reshaped

            # Create projection layer if it doesn't exist (lazy initialization)
            if self.kv_projection is None:
                input_dim = field_flat.shape[-1]
                self.kv_projection = nn.Linear(input_dim, self.d_model).to(field_state.device)
                nn.init.xavier_uniform_(self.kv_projection.weight)

            # Project all spatial positions: (N_x*N_y, D*D*2) → (N_x*N_y, d_model)
            embeddings = self.kv_projection(field_flat)  # (N_x*N_y, d_model)

        # Project to d_model (only needed for pretrained DPR path)
        if self.context_encoder is not None:
            kv_embeddings = self.kv_projection(embeddings)  # (1, d_model)
            # Expand for batch and sequence (treat as single token)
            K = kv_embeddings.unsqueeze(1)  # (1, 1, d_model)
            V = K.clone()

            # Repeat for batch
            if batch_size > 1:
                K = K.repeat(batch_size, 1, 1)  # (batch, 1, d_model)
                V = V.repeat(batch_size, 1, 1)
        else:
            # Field direct path: each spatial position becomes a token
            # embeddings: (N_x*N_y, d_model) → (1, N_x*N_y, d_model)
            K = embeddings.unsqueeze(0)  # (1, N_x*N_y, d_model)
            V = K.clone()

            # Repeat for batch
            if batch_size > 1:
                K = K.repeat(batch_size, 1, 1)  # (batch, N_x*N_y, d_model)
                V = V.repeat(batch_size, 1, 1)

        return K, V

    def generate_q(
        self,
        input_text: str,
        batch_size: int = 1
    ) -> torch.Tensor:
        """
        Generate Query vectors from input text using DPR question encoder.

        Args:
            input_text: Input question/prompt text
            batch_size: Batch size

        Returns:
            Q: Query vectors (batch, 1, d_model)
        """
        if self.question_encoder is not None:
            # Tokenize
            inputs = self.q_tokenizer(
                input_text,
                return_tensors='pt',
                padding=True,
                truncation=True,
                max_length=512
            )

            # Run DPR encoder on its own device (avoids CPU↔CUDA mismatch during compile/training)
            device = next(self.question_encoder.parameters()).device
            inputs = inputs.to(device)

            # Encode with DPR
            with torch.no_grad() if not self.question_encoder.training else torch.enable_grad():
                outputs = self.question_encoder(**inputs)
                embeddings = outputs.pooler_output  # (1, 768)
        else:
            # Fallback: random embeddings (placeholder)
            device = next(self.parameters()).device
            embeddings = torch.randn(1, self.d_model, device=device)

        # Project to d_model
        q_embeddings = embeddings if self.q_projection is None else self.q_projection(embeddings)  # (1, d_model)

        # Add sequence dimension
        Q = q_embeddings.unsqueeze(1)  # (1, 1, d_model)

        # Repeat for batch
        if batch_size > 1:
            Q = Q.repeat(batch_size, 1, 1)

        return Q

    def forward(
        self,
        field_state: torch.Tensor,
        input_text: Optional[str] = None,
        batch_size: int = 1
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Full forward pass: generate Q, K, V.

        Args:
            field_state: Cognitive tensor field (N_x, N_y, D, D)
            input_text: Optional input query text
            batch_size: Batch size

        Returns:
            K: Key vectors (batch, n_tokens, d_model)
            V: Value vectors (batch, n_tokens, d_model)
            Q: Query vectors (batch, 1, d_model) if input_text provided, else None
        """
        K, V = self.generate_kv(field_state, batch_size)

        Q = None
        if input_text is not None:
            Q = self.generate_q(input_text, batch_size)

        return K, V, Q


class DPRIntegrationConfig:
    """Configuration for DPR integration."""

    def __init__(
        self,
        d_model: int = 512,
        freeze_encoders: bool = True,
        use_pretrained: bool = True,
        enable_fine_tuning: bool = False
    ):
        """
        Initialize DPR config.

        Args:
            d_model: Model dimension for projections
            freeze_encoders: Keep DPR weights frozen
            use_pretrained: Use pre-trained DPR models
            enable_fine_tuning: If True, unfreeze encoders after initial training
        """
        self.d_model = d_model
        self.freeze_encoders = freeze_encoders
        self.use_pretrained = use_pretrained
        self.enable_fine_tuning = enable_fine_tuning

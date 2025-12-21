"""
Language Model Head

Projects transformer outputs to vocabulary logits for next-token prediction.

Standard architecture:
- Linear projection: d_model → vocab_size
- Optional: weight tying with input embeddings
- Optional: layer norm before projection
"""

import torch
import torch.nn as nn


class LanguageModelHead(nn.Module):
    """
    Language model head for next-token prediction.

    Projects hidden states to vocabulary logits.

    Args:
        d_model: Hidden dimension
        vocab_size: Vocabulary size
        use_layer_norm: Apply layer norm before projection
        tie_weights: Tie weights with input embedding (if provided)
    """

    def __init__(
        self,
        d_model: int,
        vocab_size: int,
        use_layer_norm: bool = True,
        tie_weights: bool = False,
        input_embedding: nn.Embedding = None
    ):
        super().__init__()

        self.d_model = d_model
        self.vocab_size = vocab_size
        self.tie_weights = tie_weights

        # Layer norm (optional)
        self.layer_norm = nn.LayerNorm(d_model) if use_layer_norm else nn.Identity()

        # Output projection
        if tie_weights and input_embedding is not None:
            # Weight tying: use same weights as input embedding
            self.output_projection = nn.Linear(d_model, vocab_size, bias=False)
            self.output_projection.weight = input_embedding.weight
            print("Language head: weight tying enabled")
        else:
            self.output_projection = nn.Linear(d_model, vocab_size)

        # Initialize
        if not tie_weights:
            nn.init.normal_(self.output_projection.weight, std=0.02)
            if self.output_projection.bias is not None:
                nn.init.zeros_(self.output_projection.bias)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Project to vocabulary logits.

        Args:
            hidden_states: (batch, seq_len, d_model) transformer outputs

        Returns:
            logits: (batch, seq_len, vocab_size) vocabulary logits
        """
        # Layer norm
        hidden_states = self.layer_norm(hidden_states)

        # Project to vocab
        logits = self.output_projection(hidden_states)

        return logits

"""
Supervisor Gating Module

Implements sparse attention-based expert selection.
CUDA-safe with top-k gating mechanism.
"""
try: import usage_tracker; usage_tracker.track(__file__)
except: pass

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class SupervisorGating(nn.Module):
    """
    Supervisor module for sparse expert activation.
    
    Uses top-k gating mechanism (similar to GShard) with attention-based selection.
    CUDA-Safe: Uses efficient sparse operations.
    """
    
    def __init__(self, input_dim: int, num_experts: int, top_k: int = 3, num_heads: int = 8):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.input_dim = input_dim
        
        # Gating network
        self.gate = nn.Sequential(
            nn.Linear(input_dim, num_experts * 4),
            nn.GELU(),
            nn.Linear(num_experts * 4, num_experts)
        )
        
        # Attention mechanism for dataset scanning
        self.attention = nn.MultiheadAttention(
            embed_dim=input_dim,
            num_heads=num_heads,
            batch_first=True
        )
        
    def forward(self, x: torch.Tensor, expert_embeddings: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute sparse gating scores.
        
        Args:
            x: Input (batch_size, seq_len, input_dim)
            expert_embeddings: Expert embeddings (num_experts, input_dim)
            
        Returns:
            expert_indices: Selected indices (batch_size, seq_len, top_k)
            gating_weights: Normalized weights (batch_size, seq_len, top_k)
        """
        batch_size, seq_len, _ = x.shape
        
        # Sparse dataset attention
        expert_emb_expanded = expert_embeddings.unsqueeze(0).expand(batch_size, -1, -1)
        attn_output, _ = self.attention(x, expert_emb_expanded, expert_emb_expanded)
        
        # Compute gating scores
        gate_logits = self.gate(attn_output)
        
        # Top-k selection (sparse activation)
        gating_weights, expert_indices = torch.topk(
            F.softmax(gate_logits, dim=-1), k=self.top_k, dim=-1
        )
        
        # Normalize
        gating_weights = gating_weights / (gating_weights.sum(dim=-1, keepdim=True) + 1e-8)
        
        return expert_indices, gating_weights
    
    def load_balancing_loss(self, gate_logits: torch.Tensor) -> torch.Tensor:
        """Compute load balancing loss."""
        expert_probs = F.softmax(gate_logits, dim=-1)
        usage_freq = expert_probs.mean(dim=(0, 1))
        target_freq = torch.ones_like(usage_freq) / self.num_experts
        return F.mse_loss(usage_freq, target_freq)

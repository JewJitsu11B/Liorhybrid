"""
Base Expert Module

Specialized expert modules for domain-specific processing.
CUDA-safe implementation with efficient operations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any, Tuple


class BaseExpert(nn.Module):
    """
    Base class for specialized expert modules.
    
    Each expert is trained to respond to:
    - Specific token clusters
    - Regions of the input space
    - Domain-specific knowledge
    
    CUDA-Safe: All operations use PyTorch primitives
    """
    
    def __init__(
        self,
        expert_id: int,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        specialization: str,
        dropout: float = 0.1,
    ):
        """
        Initialize expert module.
        
        Args:
            expert_id: Unique identifier for this expert
            input_dim: Input feature dimension
            hidden_dim: Hidden layer dimension
            output_dim: Output feature dimension
            specialization: Domain specialization name
            dropout: Dropout probability
        """
        super().__init__()
        self.expert_id = expert_id
        self.specialization = specialization
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # Encoder network (CUDA-compatible)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        
        # Processing layer (transformer-based)
        self.processor = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=8,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,  # Pre-norm for stability
        )
        
        # Decoder network
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )
        
        # Confidence estimation layer
        self.confidence_head = nn.Linear(hidden_dim, 1)
    
    def forward(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Process input through expert specialization.
        
        Args:
            x: Input tensor (batch_size, seq_len, input_dim)
            context: Optional context for conditioning (unused in base class)
            
        Returns:
            output: Expert output (batch_size, seq_len, output_dim)
            confidence: Expert confidence scores (batch_size, seq_len)
        """
        # Encode input
        h = self.encoder(x)  # (batch, seq, hidden)
        
        # Process with self-attention
        h = self.processor(h)  # (batch, seq, hidden)
        
        # Decode to output
        output = self.decoder(h)  # (batch, seq, output_dim)
        
        # Compute confidence scores
        confidence_logits = self.confidence_head(h)  # (batch, seq, 1)
        confidence = torch.sigmoid(confidence_logits).squeeze(-1)  # (batch, seq)
        
        return output, confidence
    
    def generate_draft_report(
        self,
        output: torch.Tensor,
        confidence: torch.Tensor
    ) -> Dict[str, Any]:
        """
        Generate compact domain-specific draft report.
        
        Args:
            output: Expert output tensor (batch_size, seq_len, output_dim)
            confidence: Confidence scores (batch_size, seq_len)
            
        Returns:
            report: Dictionary containing:
                - expert_id: Expert identifier
                - specialization: Domain specialization
                - summary: Compact summary vector (output_dim,)
                - top_k_indices: Indices of highest confidence tokens
                - avg_confidence: Average confidence score
        """
        # Summarize output (weighted mean pooling by confidence)
        weights = confidence.unsqueeze(-1)  # (batch, seq, 1)
        summary = (output * weights).sum(dim=1) / (weights.sum(dim=1) + 1e-8)
        summary = summary.mean(dim=0)  # Average across batch
        
        # Extract top-k insights (highest confidence tokens)
        k = min(10, confidence.size(1))
        # Average confidence across batch for each position
        avg_conf_per_pos = confidence.mean(dim=0)  # (seq_len,)
        top_k_idx = torch.topk(avg_conf_per_pos, k=k, dim=0).indices
        
        return {
            'expert_id': self.expert_id,
            'specialization': self.specialization,
            'summary': summary.detach(),  # (output_dim,)
            'top_k_indices': top_k_idx.detach(),  # (k,)
            'avg_confidence': confidence.mean().item(),
        }


class SpecializedExpert(BaseExpert):
    """
    Expert with additional domain-specific processing.
    
    Extends BaseExpert with custom processing for specific domains.
    """
    
    def __init__(
        self,
        expert_id: int,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        specialization: str,
        num_specialists: int = 4,
        dropout: float = 0.1,
    ):
        """
        Initialize specialized expert with sub-specialist modules.
        
        Args:
            num_specialists: Number of sub-specialist modules
        """
        super().__init__(
            expert_id, input_dim, hidden_dim, output_dim,
            specialization, dropout
        )
        
        # Sub-specialist modules for fine-grained specialization
        self.specialists = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            for _ in range(num_specialists)
        ])
        
        # Routing network for sub-specialists
        self.specialist_router = nn.Linear(hidden_dim, num_specialists)
    
    def forward(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass with sub-specialist routing."""
        # Base encoding
        h = self.encoder(x)
        h = self.processor(h)
        
        # Route to sub-specialists
        routing_weights = F.softmax(
            self.specialist_router(h), dim=-1
        )  # (batch, seq, num_specialists)
        
        # Apply sub-specialists (weighted combination)
        specialist_outputs = torch.stack([
            specialist(h) for specialist in self.specialists
        ], dim=-1)  # (batch, seq, hidden, num_specialists)
        
        # Weighted combination
        h_specialist = torch.einsum(
            'bshn,bsn->bsh',
            specialist_outputs,
            routing_weights
        )
        
        # Final decoding
        output = self.decoder(h_specialist)
        
        # Confidence (higher confidence for focused routing)
        routing_entropy = -(routing_weights * torch.log(routing_weights + 1e-8)).sum(dim=-1)
        max_entropy = torch.log(torch.tensor(len(self.specialists), device=x.device))
        confidence_logits = self.confidence_head(h_specialist)
        base_confidence = torch.sigmoid(confidence_logits).squeeze(-1)
        
        # Combine with routing sharpness (sharper routing = higher confidence)
        routing_sharpness = 1.0 - (routing_entropy / max_entropy)
        confidence = base_confidence * (0.7 + 0.3 * routing_sharpness)
        
        return output, confidence

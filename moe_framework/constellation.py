"""
Expert Constellation Coordinator

Coordinates activation of expert constellations.
CUDA-safe with efficient batched operations.
"""
try: import usage_tracker; usage_tracker.track(__file__)
except: pass

import torch
import torch.nn as nn
from typing import List, Dict, Tuple


class ExpertConstellation(nn.Module):
    """
    Coordinates expert constellation activation.
    
    A constellation is an interdependent combination of experts.
    CUDA-Safe: Efficient batched operations.
    """
    
    def __init__(self, experts: nn.ModuleList, supervisor):
        super().__init__()
        self.experts = experts
        self.supervisor = supervisor
        self.num_experts = len(experts)
        
        # Expert embeddings (learnable specialization vectors)
        self.expert_embeddings = nn.Parameter(
            torch.randn(self.num_experts, experts[0].input_dim)
        )
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[Dict]]:
        """
        Activate expert constellation.
        
        Args:
            x: Input (batch_size, seq_len, input_dim)
            
        Returns:
            combined_output: Weighted expert outputs
            draft_reports: Draft reports from activated experts
        """
        batch_size, seq_len, _ = x.shape
        
        # Supervisor determines which experts to activate
        expert_indices, gating_weights = self.supervisor(x, self.expert_embeddings)
        
        # Initialize output buffer
        combined_output = torch.zeros(
            batch_size, seq_len, self.experts[0].output_dim,
            device=x.device, dtype=x.dtype
        )
        
        # Collect draft reports
        draft_reports = []
        
        # Process each expert position
        for k in range(self.supervisor.top_k):
            # Get indices for this position
            expert_idx_k = expert_indices[:, :, k]  # (batch, seq)
            weights_k = gating_weights[:, :, k:k+1]  # (batch, seq, 1)
            
            # Process each unique expert
            unique_experts = torch.unique(expert_idx_k)
            
            for exp_id in unique_experts:
                exp_id = exp_id.item()
                
                # Mask for this expert
                mask = (expert_idx_k == exp_id)  # (batch, seq)
                
                if mask.any():
                    # Process through expert
                    output, confidence = self.experts[exp_id](x)
                    
                    # Apply weights and mask
                    weighted_output = output * weights_k
                    combined_output = combined_output + torch.where(
                        mask.unsqueeze(-1),
                        weighted_output,
                        torch.zeros_like(weighted_output)
                    )
                    
                    # Generate draft report
                    report = self.experts[exp_id].generate_draft_report(output, confidence)
                    draft_reports.append(report)
        
        return combined_output, draft_reports

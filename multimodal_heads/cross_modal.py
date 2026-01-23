"""
Cross-Modal Fusion

Fuses audio/image/text via ComplexMetricTensor.
Uses A_{mu nu} (symmetric) for semantic alignment and
B_{mu nu} (antisymmetric) for cross-modal interference.
Phase orthogonality ensures modalities don't collapse.

CUDA-safe: All operations compatible with torch.compile and CUDA graphs.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.complex_metric import ComplexMetricTensor
from models.manifold import CognitiveManifold


class CrossModalFusion(nn.Module):
    """
    Cross-modal fusion using complex metric tensor.
    
    Key physics:
    - A_{mu nu} (symmetric, Riemannian): Semantic alignment across modalities
    - B_{mu nu} (antisymmetric, symplectic): Cross-modal interference patterns
    - Phase orthogonality: Prevents mode collapse between modalities
    """
    
    def __init__(
        self,
        d_model: int = 512,
        d_coord: int = 8,
        num_modalities: int = 3,  # e.g., text, audio, image
        fusion_type: str = 'concat',  # 'concat', 'add', 'attention'
    ):
        """
        Initialize CrossModalFusion.
        
        Args:
            d_model: Model dimension
            d_coord: Coordinate manifold dimension
            num_modalities: Number of input modalities
            fusion_type: How to combine modalities
        """
        super().__init__()
        
        self.d_model = d_model
        self.num_modalities = num_modalities
        self.fusion_type = fusion_type
        
        # Complex metric for each modality
        self.complex_metrics = nn.ModuleList([
            ComplexMetricTensor(d_coord=d_coord)
            for _ in range(num_modalities)
        ])
        
        # Shared manifold for alignment
        self.shared_manifold = CognitiveManifold(
            d_embed=d_model,
            d_coord=d_coord,
            learnable_metric=True
        )
        
        # Modality-specific projections
        self.modality_projs = nn.ModuleList([
            nn.Linear(d_model, d_model)
            for _ in range(num_modalities)
        ])
        
        # Fusion layer
        if fusion_type == 'concat':
            self.fusion_proj = nn.Linear(d_model * num_modalities, d_model)
        elif fusion_type == 'attention':
            self.fusion_attention = nn.MultiheadAttention(
                d_model, num_heads=8, batch_first=True
            )
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model)
        )
    
    def compute_semantic_alignment(
        self,
        modality_embeddings: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Compute semantic alignment using Riemannian metric A.
        
        Args:
            modality_embeddings: Dict of (batch, seq, d_model) embeddings
            
        Returns:
            alignment_matrix: (num_modalities, num_modalities) alignment scores
        """
        modality_list = list(modality_embeddings.keys())
        num_mods = len(modality_list)
        
        # Project each modality to shared manifold
        coords_list = []
        for i, mod_name in enumerate(modality_list):
            emb = modality_embeddings[mod_name]
            coords, _ = self.shared_manifold.project(emb)
            
            # Pool to single vector per modality
            coords_pooled = coords.mean(dim=1)  # (batch, d_coord)
            coords_list.append(coords_pooled)
        
        # Compute alignment via geodesic distances
        alignment = torch.zeros(num_mods, num_mods, device=coords_list[0].device)
        
        for i in range(num_mods):
            for j in range(num_mods):
                if i != j:
                    # Geodesic distance on manifold
                    dist = self.shared_manifold.geodesic_distance(
                        coords_list[i], coords_list[j]
                    )
                    # Convert to similarity (closer = more aligned)
                    alignment[i, j] = torch.exp(-dist.mean())
        
        return alignment
    
    def compute_phase_interference(
        self,
        modality_embeddings: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Compute cross-modal interference using symplectic form B.
        
        Args:
            modality_embeddings: Dict of (batch, seq, d_model) embeddings
            
        Returns:
            interference: (batch, seq, d_model) interference pattern
        """
        modality_list = list(modality_embeddings.keys())
        
        # Compute phase fields for each modality
        phase_fields = []
        alpha = torch.tensor(0.5, device=list(modality_embeddings.values())[0].device)
        
        for i, mod_name in enumerate(modality_list):
            emb = modality_embeddings[mod_name]
            phase = self.complex_metrics[i].compute_phase_field(emb, alpha)
            phase_fields.append(phase)
        
        # Compute interference via phase differences
        interference = torch.zeros_like(list(modality_embeddings.values())[0])
        
        for i in range(len(phase_fields)):
            for j in range(i+1, len(phase_fields)):
                # Phase difference creates interference
                phase_diff = phase_fields[i] - phase_fields[j]
                interference_ij = torch.sin(phase_diff).unsqueeze(-1)
                
                # Weight by both modalities
                emb_i = list(modality_embeddings.values())[i]
                emb_j = list(modality_embeddings.values())[j]
                interference = interference + interference_ij * (emb_i + emb_j) / 2
        
        return interference
    
    def forward(
        self,
        modality_embeddings: Dict[str, torch.Tensor],
        return_alignment: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Fuse multiple modalities.
        
        Args:
            modality_embeddings: Dict mapping modality names to embeddings
                                Each (batch, seq, d_model)
            return_alignment: Whether to return alignment matrix
            
        Returns:
            fused: (batch, seq, d_model) fused embeddings
            alignment: Optional alignment matrix
        """
        # Project each modality
        projected = {}
        for i, (mod_name, emb) in enumerate(modality_embeddings.items()):
            projected[mod_name] = self.modality_projs[i](emb)
        
        # Compute semantic alignment (A metric)
        alignment = self.compute_semantic_alignment(projected)
        
        # Compute phase interference (B metric)
        interference = self.compute_phase_interference(projected)
        
        # Fusion
        if self.fusion_type == 'concat':
            # Concatenate all modalities
            concat_list = [emb for emb in projected.values()]
            # Ensure same sequence length (pad if needed)
            max_seq = max(emb.shape[1] for emb in concat_list)
            padded = []
            for emb in concat_list:
                if emb.shape[1] < max_seq:
                    pad_len = max_seq - emb.shape[1]
                    emb = F.pad(emb, (0, 0, 0, pad_len))
                padded.append(emb)
            
            fused = torch.cat(padded, dim=-1)
            fused = self.fusion_proj(fused)
            
        elif self.fusion_type == 'add':
            # Simple addition (requires same seq length)
            fused = sum(projected.values()) / len(projected)
            
        elif self.fusion_type == 'attention':
            # Cross-attention between modalities
            # Use first modality as query, others as key/value
            mod_list = list(projected.values())
            query = mod_list[0]
            key_value = torch.cat(mod_list[1:], dim=1)
            
            fused, _ = self.fusion_attention(query, key_value, key_value)
        
        # Add interference pattern
        fused = fused + interference
        
        # Output projection
        output = self.output_proj(fused)
        
        if return_alignment:
            return output, alignment
        else:
            return output, None

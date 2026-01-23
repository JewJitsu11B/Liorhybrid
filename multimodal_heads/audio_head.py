"""
Audio Causal Head

Uses LiorKernel for temporal memory + ComplexMetricTensor for phase structure.
Processes audio spectrograms with phase orthogonality for frequency stability
and power-law memory for long-range audio dependencies.

CUDA-safe: All operations compatible with torch.compile and CUDA graphs.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.lior_kernel import LiorKernel, LiorMemoryState
from models.complex_metric import ComplexMetricTensor
from models.causal_field import CausalFieldLayer


class AudioCausalHead(nn.Module):
    """
    Audio processing head using physics framework.
    
    Pipeline:
    1. Spectrogram input → ComplexMetricTensor (phase field from frequency)
    2. LiorKernel O(1) temporal recurrence across time axis
    3. CausalFieldLayer for parallel processing (replaces RNN)
    4. Output audio embeddings or generation logits
    
    Key physics:
    - Phase orthogonality ensures frequency stability
    - Power-law memory captures long-range dependencies
    - Complex metric separates semantic (A) from phase (B)
    """
    
    def __init__(
        self,
        n_mels: int = 80,
        d_model: int = 512,
        d_field: int = 16,
        kernel_size: int = 64,
        output_type: str = 'embedding',  # 'embedding' or 'generation'
        vocab_size: Optional[int] = None,
    ):
        """
        Initialize AudioCausalHead.
        
        Args:
            n_mels: Number of mel frequency bins
            d_model: Model dimension
            d_field: Field dimension for causal layer
            kernel_size: LiorKernel size
            output_type: 'embedding' for features or 'generation' for logits
            vocab_size: Size of vocabulary for generation (required if output_type='generation')
        """
        super().__init__()
        
        self.n_mels = n_mels
        self.d_model = d_model
        self.output_type = output_type
        
        # Input projection from mel spectrogram
        self.input_proj = nn.Linear(n_mels, d_model)
        
        # Complex metric for phase structure
        self.complex_metric = ComplexMetricTensor(d_coord=8)
        
        # LiorKernel for O(1) temporal memory
        self.lior_kernel = LiorKernel(
            p_eff=4,
            init_tau_exp=1.0,
            init_tau_frac=10.0,  # Longer for audio dependencies
            init_tau_osc=2.0,
        )
        
        # Causal field layer for parallel processing
        self.causal_layer = CausalFieldLayer(
            d_model=d_model,
            d_field=d_field,
            d_spinor=4,
            kernel_size=kernel_size
        )
        
        # Output projection
        if output_type == 'embedding':
            self.output_proj = nn.Sequential(
                nn.LayerNorm(d_model),
                nn.Linear(d_model, d_model)
            )
        elif output_type == 'generation':
            assert vocab_size is not None, "vocab_size required for generation"
            self.output_proj = nn.Sequential(
                nn.LayerNorm(d_model),
                nn.Linear(d_model, vocab_size)
            )
        else:
            raise ValueError(f"Unknown output_type: {output_type}")
        
        # Memory state for LiorKernel
        self.memory_state = None
    
    def reset_memory(self):
        """Reset LiorKernel memory state."""
        self.memory_state = None
    
    def forward(
        self,
        spectrogram: torch.Tensor,
        lengths: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through audio causal head.
        
        Args:
            spectrogram: (batch, time, n_mels) audio spectrogram
            lengths: (batch,) optional sequence lengths for masking
            
        Returns:
            output: (batch, time, d_model) embeddings or (batch, time, vocab_size) logits
            phase_field: (batch, time) phase field from complex metric
        """
        batch_size, time_steps, _ = spectrogram.shape
        
        # Project spectrogram to model dimension
        x = self.input_proj(spectrogram)  # (batch, time, d_model)
        
        # Compute phase field from frequency structure
        # Use complex metric to extract phase information
        alpha = torch.tensor(0.5, device=x.device)  # Fractional order
        phase_field = self.complex_metric.compute_phase_field(x, alpha)
        
        # Apply phase modulation (A + iB structure)
        # A (Riemannian): semantic content
        # B (symplectic): phase/frequency structure
        phase_mod = torch.exp(1j * phase_field.unsqueeze(-1))
        x_complex = x.to(torch.complex64) * phase_mod
        x = x_complex.real  # Use real part for processing
        
        # LiorKernel O(1) temporal recurrence
        # Initialize memory state if needed
        if self.memory_state is None or self.memory_state.m.size(0) != batch_size:
            self.memory_state = LiorMemoryState.initialize(
                batch_size, self.d_model, self.lior_kernel.p_eff, device=x.device
            )
        
        # Apply recurrence across time
        x_recurrent = []
        for t in range(time_steps):
            x_t = x[:, t, :]  # (batch, d_model)
            
            # LiorKernel update
            m_new = self.lior_kernel.recurrence_step(
                x_t, self.memory_state.m, self.memory_state.x_history
            )
            
            # Update state
            self.memory_state = self.memory_state.update(x_t, m_new)
            
            # Combine input with memory
            x_t_combined = x_t + m_new
            x_recurrent.append(x_t_combined)
        
        x = torch.stack(x_recurrent, dim=1)  # (batch, time, d_model)
        
        # Causal field layer for parallel processing
        x = self.causal_layer(x)
        
        # Output projection
        output = self.output_proj(x)
        
        return output, phase_field


class AudioEncoder(nn.Module):
    """
    Complete audio encoder using AudioCausalHead.
    
    Wrapper for easy integration with existing pipelines.
    """
    
    def __init__(
        self,
        n_mels: int = 80,
        d_model: int = 512,
        num_layers: int = 6,
    ):
        super().__init__()
        
        self.layers = nn.ModuleList([
            AudioCausalHead(
                n_mels=n_mels if i == 0 else d_model,
                d_model=d_model,
                output_type='embedding'
            )
            for i in range(num_layers)
        ])
        
    def forward(self, spectrogram: torch.Tensor) -> torch.Tensor:
        """Encode audio spectrogram to embeddings."""
        x = spectrogram
        
        for layer in self.layers:
            x, _ = layer(x)
            layer.reset_memory()
        
        return x

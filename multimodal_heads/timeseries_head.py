"""
Time Series Head

Uses full LiorKernel capabilities for time series processing.
- Exponential mode: Short-term patterns
- Power-law mode: Long-range dependencies
- Oscillatory mode: Periodic/seasonal effects
- O(1) recurrence: Real-time streaming

CUDA-safe: All operations compatible with torch.compile and CUDA graphs.
"""
try: import usage_tracker; usage_tracker.track(__file__)
except: pass

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.lior_kernel import LiorKernel, LiorMemoryState


class TimeSeriesHead(nn.Module):
    """
    Time series processing using full LiorKernel.
    
    Key physics:
    - Exponential mode: Captures short-term Markovian patterns
    - Power-law mode: Captures long-range fractional dependencies
    - Oscillatory mode: Captures periodic and seasonal patterns
    - O(1) recurrence: Enables real-time streaming processing
    """
    
    def __init__(
        self,
        input_dim: int,
        d_model: int = 512,
        output_dim: Optional[int] = None,
        init_tau_exp: float = 1.0,
        init_tau_frac: float = 10.0,
        init_tau_osc: float = 5.0,
        init_omega: float = 0.5,  # Oscillation frequency
        p_eff: int = 4,
    ):
        """
        Initialize TimeSeriesHead.
        
        Args:
            input_dim: Input feature dimension
            d_model: Hidden dimension
            output_dim: Output dimension (defaults to input_dim)
            init_tau_exp: Exponential mode timescale
            init_tau_frac: Fractional mode timescale
            init_tau_osc: Oscillatory mode timescale
            init_omega: Oscillation frequency for periodic patterns
            p_eff: Effective pole count for recurrence
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.d_model = d_model
        self.output_dim = output_dim or input_dim
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, d_model)
        
        # LiorKernel with all three modes
        self.lior_kernel = LiorKernel(
            p_eff=p_eff,
            init_tau_exp=init_tau_exp,
            init_tau_frac=init_tau_frac,
            init_tau_osc=init_tau_osc,
            init_alpha=0.7,  # Fractional order
            init_omega=init_omega,
            init_phi=0.0,
        )
        
        # Processing layers
        self.process_layer = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model)
        )
        
        # Output projection
        self.output_proj = nn.Linear(d_model, self.output_dim)
        
        # Memory state
        self.memory_state = None
        
    def reset_memory(self):
        """Reset memory state for new sequence."""
        self.memory_state = None
    
    def forward(
        self,
        x: torch.Tensor,
        return_memory: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass for time series.
        
        Args:
            x: (batch, seq_len, input_dim) input time series
            return_memory: Whether to return memory state
            
        Returns:
            output: (batch, seq_len, output_dim) processed time series
            memory: Optional (batch, d_model) final memory state
        """
        batch_size, seq_len, _ = x.shape
        
        # Project input
        x = self.input_proj(x)  # (batch, seq_len, d_model)
        
        # Initialize memory if needed
        if self.memory_state is None or self.memory_state.m.size(0) != batch_size:
            self.memory_state = LiorMemoryState.initialize(
                batch_size, self.d_model, self.lior_kernel.p_eff, device=x.device
            )
        
        # Process sequence with O(1) recurrence
        outputs = []
        for t in range(seq_len):
            x_t = x[:, t, :]  # (batch, d_model)
            
            # LiorKernel recurrence update
            m_new = self.lior_kernel.recurrence_step(
                x_t, self.memory_state.m, self.memory_state.x_history
            )
            
            # Update memory state
            self.memory_state = self.memory_state.update(x_t, m_new)
            
            # Combine input with memory
            x_t_combined = x_t + m_new
            
            # Process
            x_t_processed = self.process_layer(x_t_combined)
            
            # Residual
            x_t_out = x_t + x_t_processed
            
            outputs.append(x_t_out)
        
        output = torch.stack(outputs, dim=1)  # (batch, seq_len, d_model)
        
        # Output projection
        output = self.output_proj(output)
        
        if return_memory:
            return output, self.memory_state.m
        else:
            return output, None
    
    def forecast(
        self,
        x: torch.Tensor,
        horizon: int
    ) -> torch.Tensor:
        """
        Forecast future values.
        
        Args:
            x: (batch, seq_len, input_dim) historical time series
            horizon: Number of steps to forecast
            
        Returns:
            forecast: (batch, horizon, output_dim) forecasted values
        """
        batch_size = x.shape[0]
        
        # Process historical data to build memory
        _, _ = self.forward(x)
        
        # Generate forecasts autoregressively
        forecasts = []
        last_input = x[:, -1:, :]  # (batch, 1, input_dim)
        
        for _ in range(horizon):
            # Project to d_model
            x_proj = self.input_proj(last_input.squeeze(1))  # (batch, d_model)
            
            # Recurrence step
            m_new = self.lior_kernel.recurrence_step(
                x_proj, self.memory_state.m, self.memory_state.x_history
            )
            
            # Update memory
            self.memory_state = self.memory_state.update(x_proj, m_new)
            
            # Process
            x_combined = x_proj + m_new
            x_processed = self.process_layer(x_combined)
            x_out = x_proj + x_processed
            
            # Output
            pred = self.output_proj(x_out)  # (batch, output_dim)
            forecasts.append(pred)
            
            # Use prediction as next input (if output_dim matches input_dim)
            if self.output_dim == self.input_dim:
                last_input = pred.unsqueeze(1)
            else:
                # Otherwise use last real input (teacher forcing disabled)
                last_input = last_input
        
        forecast = torch.stack(forecasts, dim=1)  # (batch, horizon, output_dim)
        
        return forecast

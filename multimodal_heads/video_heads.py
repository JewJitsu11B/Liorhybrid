"""
Multispectral Video Heads

Specialized heads for IR (infrared), visible, and UV (ultraviolet) video processing.
Each can work independently or fuse together using physics-based spectral alignment.

Uses ComplexMetricTensor for spectral phase structure,
LiorKernel for temporal dynamics,
CognitiveManifold for spatial-spectral geometry.

CUDA-safe: All operations compatible with torch.compile and CUDA graphs.
"""
try: import usage_tracker; usage_tracker.track(__file__)
except: pass

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, List

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.lior_kernel import LiorKernel, LiorMemoryState
from models.complex_metric import ComplexMetricTensor, SpinorBilinears
from models.manifold import CognitiveManifold
from models.causal_field import CausalFieldLayer


class SpectralVideoHead(nn.Module):
    """
    Base class for spectral video processing.
    
    Handles spatial-temporal processing with spectral-specific phase structure.
    """
    
    def __init__(
        self,
        spectrum_type: str,  # 'IR', 'visible', 'UV'
        in_channels: int,
        d_model: int = 512,
        d_coord: int = 8,
        img_size: int = 224,
        patch_size: int = 16,
        spectral_bands: int = 1,  # Number of spectral bands
    ):
        """
        Initialize SpectralVideoHead.
        
        Args:
            spectrum_type: Type of spectrum ('IR', 'visible', 'UV')
            in_channels: Input channels (e.g., 1 for IR, 3 for RGB, 1 for UV)
            d_model: Model dimension
            d_coord: Coordinate manifold dimension
            img_size: Frame size
            patch_size: Spatial patch size
            spectral_bands: Number of spectral bands to process
        """
        super().__init__()
        
        self.spectrum_type = spectrum_type
        self.in_channels = in_channels
        self.d_model = d_model
        self.spectral_bands = spectral_bands
        
        # Spectral-specific wavelength ranges (in nm)
        self.wavelength_ranges = {
            'IR': (700, 2500),      # Infrared: 700-2500nm
            'visible': (380, 700),  # Visible: 380-700nm
            'UV': (10, 380),        # Ultraviolet: 10-380nm
        }
        
        self.wavelength_min, self.wavelength_max = self.wavelength_ranges[spectrum_type]
        
        # Calculate patches
        self.n_patches = (img_size // patch_size) ** 2
        self.patch_dim = in_channels * patch_size * patch_size
        
        # Patch embedding with spectral encoding
        self.patch_embed = nn.Linear(self.patch_dim, d_model)
        
        # Spectral encoding (wavelength-dependent)
        self.spectral_encoder = nn.Sequential(
            nn.Linear(spectral_bands, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, d_model)
        )
        
        # Positional embedding
        self.pos_embed = nn.Parameter(torch.randn(1, self.n_patches, d_model) * 0.02)
        
        # Temporal encoding
        self.temporal_embed = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        
        # Complex metric for spectral phase structure
        self.complex_metric = ComplexMetricTensor(d_coord=d_coord)
        
        # LiorKernel for temporal dynamics
        self.lior_kernel = LiorKernel(
            p_eff=4,
            init_tau_exp=0.5,   # Fast for video
            init_tau_frac=5.0,  # Medium range
            init_tau_osc=2.0,
        )
        
        # Cognitive manifold for spatial-spectral geometry
        self.manifold = CognitiveManifold(
            d_embed=d_model,
            d_coord=d_coord,
            learnable_metric=True
        )
        
        # Causal field for spatial processing
        self.spatial_layer = CausalFieldLayer(
            d_model=d_model,
            d_field=16,
            d_spinor=4,
            kernel_size=64
        )
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model)
        )
        
        # Memory state for temporal processing
        self.memory_state = None
    
    def reset_memory(self):
        """Reset temporal memory."""
        self.memory_state = None
    
    def compute_spectral_phase(
        self,
        x: torch.Tensor,
        wavelengths: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute spectral phase field based on wavelength.
        
        Args:
            x: (batch, n_patches, d_model) embeddings
            wavelengths: Optional (spectral_bands,) wavelength values
            
        Returns:
            phase: (batch, n_patches) spectral phase field
        """
        # Use wavelength range to compute fractional order
        # Shorter wavelength (UV) = higher frequency = higher alpha
        # Longer wavelength (IR) = lower frequency = lower alpha
        wavelength_center = (self.wavelength_min + self.wavelength_max) / 2
        
        # Map wavelength to alpha: UV (short) -> high alpha, IR (long) -> low alpha
        # Normalize to [0.3, 0.7] range
        alpha_value = 0.7 - ((wavelength_center - 10) / (2500 - 10)) * 0.4
        alpha = torch.tensor(alpha_value, device=x.device)
        
        # Compute phase field using complex metric
        phase = self.complex_metric.compute_phase_field(x, alpha)
        
        return phase
    
    def patchify(self, frames: torch.Tensor) -> torch.Tensor:
        """
        Convert video frames to patches.
        
        Args:
            frames: (batch, time, C, H, W) video frames
            
        Returns:
            patches: (batch, time, n_patches, patch_dim) flattened patches
        """
        batch_size, time_steps, C, H, W = frames.shape
        
        # Process each frame
        patches_list = []
        for t in range(time_steps):
            frame = frames[:, t]  # (batch, C, H, W)
            
            # Unfold to patches
            patches = F.unfold(
                frame,
                kernel_size=self.patch_embed.weight.shape[1] // C,  # patch_size
                stride=self.patch_embed.weight.shape[1] // C
            )  # (batch, patch_dim, n_patches)
            
            patches = patches.transpose(1, 2)  # (batch, n_patches, patch_dim)
            patches_list.append(patches)
        
        patches = torch.stack(patches_list, dim=1)  # (batch, time, n_patches, patch_dim)
        
        return patches
    
    def forward(
        self,
        video: torch.Tensor,
        wavelengths: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for spectral video.
        
        Args:
            video: (batch, time, C, H, W) video frames
            wavelengths: Optional (spectral_bands,) wavelength encoding
            
        Returns:
            embeddings: (batch, time, n_patches, d_model) spatial-temporal embeddings
            phase: (batch, time, n_patches) spectral phase field
        """
        batch_size, time_steps, _, _, _ = video.shape
        
        # Patchify
        patches = self.patchify(video)  # (batch, time, n_patches, patch_dim)
        
        # Patch embedding
        x = self.patch_embed(patches)  # (batch, time, n_patches, d_model)
        
        # Add positional embedding (spatial)
        x = x + self.pos_embed.unsqueeze(1)
        
        # Add spectral encoding if wavelengths provided
        if wavelengths is not None:
            spectral_enc = self.spectral_encoder(wavelengths)
            x = x + spectral_enc.view(1, 1, 1, -1)
        
        # Compute spectral phase
        phase_list = []
        x_processed_list = []
        
        # Initialize memory if needed
        if self.memory_state is None or self.memory_state.m.size(0) != batch_size * self.n_patches:
            self.memory_state = LiorMemoryState.initialize(
                batch_size * self.n_patches, self.d_model,
                self.lior_kernel.p_eff, device=x.device
            )
        
        # Process temporally with LiorKernel
        for t in range(time_steps):
            x_t = x[:, t]  # (batch, n_patches, d_model)
            
            # Compute spectral phase
            phase_t = self.compute_spectral_phase(x_t, wavelengths)
            phase_list.append(phase_t)
            
            # Apply phase modulation
            phase_mod = torch.exp(1j * phase_t.unsqueeze(-1))
            x_t_complex = x_t.to(torch.complex64) * phase_mod
            x_t = x_t_complex.real
            
            # Flatten for temporal processing
            x_t_flat = x_t.reshape(batch_size * self.n_patches, self.d_model)
            
            # LiorKernel temporal recurrence
            m_new = self.lior_kernel.recurrence_step(
                x_t_flat, self.memory_state.m, self.memory_state.x_history
            )
            self.memory_state = self.memory_state.update(x_t_flat, m_new)
            
            # Combine with memory
            x_t_flat = x_t_flat + m_new
            x_t = x_t_flat.reshape(batch_size, self.n_patches, self.d_model)
            
            # Spatial processing with causal field
            x_t = self.spatial_layer(x_t)
            
            # Add temporal embedding
            x_t = x_t + self.temporal_embed * t / time_steps
            
            x_processed_list.append(x_t)
        
        x_processed = torch.stack(x_processed_list, dim=1)  # (batch, time, n_patches, d_model)
        phase = torch.stack(phase_list, dim=1)  # (batch, time, n_patches)
        
        # Output projection
        output = self.output_proj(x_processed)
        
        return output, phase


class IRVideoHead(SpectralVideoHead):
    """Infrared video processing head (700-2500nm)."""
    
    def __init__(
        self,
        d_model: int = 512,
        img_size: int = 224,
        patch_size: int = 16,
    ):
        super().__init__(
            spectrum_type='IR',
            in_channels=1,  # Single channel for IR
            d_model=d_model,
            img_size=img_size,
            patch_size=patch_size,
            spectral_bands=1
        )


class VisibleVideoHead(SpectralVideoHead):
    """Visible spectrum video processing head (380-700nm)."""
    
    def __init__(
        self,
        d_model: int = 512,
        img_size: int = 224,
        patch_size: int = 16,
    ):
        super().__init__(
            spectrum_type='visible',
            in_channels=3,  # RGB channels
            d_model=d_model,
            img_size=img_size,
            patch_size=patch_size,
            spectral_bands=3
        )


class UVVideoHead(SpectralVideoHead):
    """Ultraviolet video processing head (10-380nm)."""
    
    def __init__(
        self,
        d_model: int = 512,
        img_size: int = 224,
        patch_size: int = 16,
    ):
        super().__init__(
            spectrum_type='UV',
            in_channels=1,  # Single channel for UV
            d_model=d_model,
            img_size=img_size,
            patch_size=patch_size,
            spectral_bands=1
        )


class MultispectralVideoFusion(nn.Module):
    """
    Fuses IR, visible, and UV video streams.
    
    Uses complex metric for spectral alignment:
    - A (Riemannian): Spatial-temporal structure
    - B (symplectic): Spectral phase interference
    """
    
    def __init__(
        self,
        d_model: int = 512,
        d_coord: int = 8,
        fusion_type: str = 'attention',  # 'concat', 'add', 'attention'
    ):
        """
        Initialize MultispectralVideoFusion.
        
        Args:
            d_model: Model dimension
            d_coord: Coordinate manifold dimension
            fusion_type: Fusion strategy
        """
        super().__init__()
        
        self.d_model = d_model
        self.fusion_type = fusion_type
        
        # Complex metrics for each spectrum
        self.ir_metric = ComplexMetricTensor(d_coord=d_coord)
        self.visible_metric = ComplexMetricTensor(d_coord=d_coord)
        self.uv_metric = ComplexMetricTensor(d_coord=d_coord)
        
        # Shared manifold for alignment
        self.shared_manifold = CognitiveManifold(
            d_embed=d_model,
            d_coord=d_coord,
            learnable_metric=True
        )
        
        # Spectral alignment layers
        self.ir_align = nn.Linear(d_model, d_model)
        self.visible_align = nn.Linear(d_model, d_model)
        self.uv_align = nn.Linear(d_model, d_model)
        
        # Fusion layer
        if fusion_type == 'concat':
            self.fusion_proj = nn.Linear(d_model * 3, d_model)
        elif fusion_type == 'attention':
            self.cross_attention = nn.MultiheadAttention(
                d_model, num_heads=8, batch_first=True
            )
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model)
        )
    
    def compute_spectral_alignment(
        self,
        ir_emb: torch.Tensor,
        visible_emb: torch.Tensor,
        uv_emb: torch.Tensor,
        ir_phase: torch.Tensor,
        visible_phase: torch.Tensor,
        uv_phase: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute spectral alignment via phase coherence.
        
        Returns:
            alignment: (3, 3) alignment matrix
        """
        # Compute phase coherence between spectra
        alignment = torch.zeros(3, 3, device=ir_emb.device)
        
        phases = [ir_phase, visible_phase, uv_phase]
        embeddings = [ir_emb, visible_emb, uv_emb]
        
        for i in range(3):
            for j in range(3):
                if i != j:
                    # Phase difference indicates spectral relationship
                    phase_diff = (phases[i] - phases[j]).abs().mean()
                    # Convert to similarity
                    alignment[i, j] = torch.exp(-phase_diff)
        
        return alignment
    
    def forward(
        self,
        ir_emb: Optional[torch.Tensor] = None,
        visible_emb: Optional[torch.Tensor] = None,
        uv_emb: Optional[torch.Tensor] = None,
        ir_phase: Optional[torch.Tensor] = None,
        visible_phase: Optional[torch.Tensor] = None,
        uv_phase: Optional[torch.Tensor] = None,
        return_alignment: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Fuse multispectral video embeddings.
        
        Args:
            ir_emb: Optional (batch, time, n_patches, d_model) IR embeddings
            visible_emb: Optional (batch, time, n_patches, d_model) visible embeddings
            uv_emb: Optional (batch, time, n_patches, d_model) UV embeddings
            ir_phase, visible_phase, uv_phase: Corresponding phase fields
            return_alignment: Whether to return alignment matrix
            
        Returns:
            fused: (batch, time, n_patches, d_model) fused embeddings
            alignment: Optional (3, 3) alignment matrix
        """
        available_spectra = []
        aligned_embeddings = []
        phases = []
        
        # Align each available spectrum
        if ir_emb is not None:
            ir_aligned = self.ir_align(ir_emb)
            available_spectra.append('IR')
            aligned_embeddings.append(ir_aligned)
            if ir_phase is not None:
                phases.append(ir_phase)
        
        if visible_emb is not None:
            visible_aligned = self.visible_align(visible_emb)
            available_spectra.append('visible')
            aligned_embeddings.append(visible_aligned)
            if visible_phase is not None:
                phases.append(visible_phase)
        
        if uv_emb is not None:
            uv_aligned = self.uv_align(uv_emb)
            available_spectra.append('UV')
            aligned_embeddings.append(uv_aligned)
            if uv_phase is not None:
                phases.append(uv_phase)
        
        if not aligned_embeddings:
            raise ValueError("At least one spectrum must be provided")
        
        # Compute alignment if phases available
        alignment = None
        if len(phases) >= 2 and return_alignment:
            # Pad to 3 phases if needed
            while len(phases) < 3:
                phases.append(torch.zeros_like(phases[0]))
            
            alignment = self.compute_spectral_alignment(
                aligned_embeddings[0] if len(aligned_embeddings) > 0 else torch.zeros_like(aligned_embeddings[0]),
                aligned_embeddings[1] if len(aligned_embeddings) > 1 else torch.zeros_like(aligned_embeddings[0]),
                aligned_embeddings[2] if len(aligned_embeddings) > 2 else torch.zeros_like(aligned_embeddings[0]),
                phases[0], phases[1], phases[2]
            )
        
        # Fusion
        if len(aligned_embeddings) == 1:
            # Single spectrum, no fusion needed
            fused = aligned_embeddings[0]
        elif self.fusion_type == 'concat':
            # Concatenate all spectra
            fused = torch.cat(aligned_embeddings, dim=-1)
            fused = self.fusion_proj(fused)
        elif self.fusion_type == 'add':
            # Simple average
            fused = torch.stack(aligned_embeddings, dim=0).mean(dim=0)
        elif self.fusion_type == 'attention':
            # Cross-attention fusion
            # Use first as query, others as key/value
            batch, time, n_patches, d_model = aligned_embeddings[0].shape
            
            # Reshape for attention
            query = aligned_embeddings[0].reshape(batch * time, n_patches, d_model)
            
            if len(aligned_embeddings) > 1:
                key_value = torch.cat([
                    emb.reshape(batch * time, n_patches, d_model)
                    for emb in aligned_embeddings[1:]
                ], dim=1)
                
                fused, _ = self.cross_attention(query, key_value, key_value)
                fused = fused.reshape(batch, time, n_patches, d_model)
            else:
                fused = aligned_embeddings[0]
        
        # Output projection
        output = self.output_proj(fused)
        
        if return_alignment:
            return output, alignment
        else:
            return output, None

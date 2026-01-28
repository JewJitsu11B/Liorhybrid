"""
Configuration and Parameters for Bayesian Cognitive Field

This module defines all parameters from the paper (see Table in Section 7).

References:
- Paper Equation (1): Master equation
- Paper Table 1: Parameter reference
"""
try: import usage_tracker; usage_tracker.track(__file__)
except: pass

from dataclasses import dataclass
import torch


@dataclass
class FieldConfig:
    """
    Configuration for cognitive tensor field evolution.

    All parameters correspond to symbols in the paper:
    - hbar_cog: ℏ_cog (Equation 1)
    - m_cog: m_cog (Equation 2)
    - lambda_QR: λ_QR (Equation 4)
    - lambda_F: λ_F (Equation 7)
    - alpha: α (Equation 8, fractional order)
    - tau: τ (Equation 6, Bayesian temperature)
    - nu: ν(x) (Definition 2, spatial entropy variation)
    - adaptive_learning: Enable Corollary (Adaptive Learning)
    """

    # Spatial dimensions
    spatial_size: tuple = (28, 28)  # N_x × N_y grid
    tensor_dim: int = 16            # D ≥ 16 (internal cognitive space)

    # Physical parameters (from paper Table 1)
    hbar_cog: float = 0.1           # Cognitive Planck constant
    m_cog: float = 1.0              # Cognitive effective mass

    # Operator strengths
    lambda_QR: float = 0.3          # Bayesian update gain (0.1 - 0.5)
    lambda_F: float = 0.05          # Memory damping strength (0.01 - 0.1)

    # Fractional memory
    alpha: float = 0.5              # Fractional order (0.3 - 0.7)
    memory_window: int = 20         # N_mem history buffer size

    # Bayesian parameters
    tau: float = 0.5                # Temperature for updates (0.1 - 1.0)
    nu: float = 0.5                 # Spatial entropy variation (0 - 1.0)

    # Adaptive learning (Paper Corollary: Adaptive Learning)
    adaptive_learning: bool = False # Enable parameter adaptation
    param_learning_rate: float = 0.001  # Learning rate for gradient descent

    # Integration
    dt: float = 0.005               # Timestep (0.001 - 0.01)

    # Device
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    dtype: torch.dtype = torch.complex64

    def __post_init__(self):
        """Validate parameters against stability constraints."""
        # CFL-like condition (paper Section 5.1)
        dx = 1.0  # Assuming unit grid spacing
        max_dt = (2 * self.m_cog * dx**2) / self.hbar_cog**2

        if self.dt > max_dt:
            raise ValueError(
                f"Timestep dt={self.dt} exceeds stability limit {max_dt:.4f}. "
                f"See paper Equation (13) for CFL condition."
            )

        if self.tensor_dim < 16:
            import warnings
            warnings.warn(
                f"tensor_dim={self.tensor_dim} < 16 may have insufficient DOF "
                f"for overdetermination. See paper Implementation Note 1."
            )


def get_default_config() -> FieldConfig:
    """Returns validated default configuration."""
    return FieldConfig()


# Pre-defined configurations for common tasks

MNIST_CONFIG = FieldConfig(
    spatial_size=(28, 28),
    tensor_dim=16,
    hbar_cog=0.1,
    lambda_QR=0.3,
    lambda_F=0.05,
    alpha=0.5,
    tau=0.3,
    dt=0.005
)

FAST_TEST_CONFIG = FieldConfig(
    spatial_size=(8, 8),  # Smaller for quick tests
    tensor_dim=16,  # Minimum 16 for sufficient DOF
    hbar_cog=0.1,
    lambda_QR=0.2,
    lambda_F=0.03,
    alpha=0.4,
    memory_window=10,
    tau=0.5,
    dt=0.01
)


# =============================================================================
# Training Configuration
# =============================================================================

@dataclass
class TrainingConfig:
    """
    Unified configuration for all training hyperparameters.

    This centralizes all ML/training parameters to ensure consistency
    across the codebase. Use preset configurations or customize as needed.

    Attributes:
        Model Architecture:
            d_model: Hidden dimension size
            n_heads: Number of attention heads
            n_layers: Number of transformer/mamba layers
            d_ff: Feed-forward hidden dimension (default: 4 * d_model)
            dropout: Dropout rate

        Mamba-specific:
            use_mamba: Use Mamba architecture instead of standard transformer
            n_mamba_layers: Number of Mamba layers (if use_mamba=True)
            n_attention_layers: Number of attention layers in hybrid mode

        Training:
            batch_size: Training batch size
            max_epochs: Maximum training epochs
            learning_rate: Initial learning rate
            weight_decay: L2 regularization strength
            warmup_steps: Linear warmup steps for learning rate
            grad_accum_steps: Gradient accumulation steps
            clip_grad_norm: Maximum gradient norm for clipping

        Data:
            max_seq_len: Maximum sequence length
            vocab_size: Vocabulary size for tokenizer
            val_split: Validation set fraction
            test_split: Test set fraction

        Field:
            field_dim: Tensor field dimension (should be >= 16)
            spatial_size: Field spatial grid size [H, W]
            adaptive_field: Enable adaptive field evolution

        Optimization:
            use_amp: Use automatic mixed precision
            patience: Early stopping patience (epochs)

        Logging:
            log_interval: Steps between logging
            eval_interval: Steps between evaluation
            save_interval: Steps between checkpoints
            output_dir: Directory for checkpoints and logs
    """

    # === Model Architecture ===
    d_model: int = 512
    n_heads: int = 8
    n_layers: int = 6
    d_ff: int = None  # Deprecated: use ffn_expansion_factor instead
    dropout: float = 0.1
    ffn_activation: str = 'swiglu'  # 'swiglu' (default), 'gelu', 'relu'
    ffn_expansion_factor: float = None  # None -> 8/3 for SwiGLU, 4 for legacy

    # === Mamba Configuration ===
    use_mamba: bool = True
    n_mamba_layers: int = 4
    n_attention_layers: int = 2

    # === Training Hyperparameters ===
    batch_size: int = 64
    max_epochs: int = 10
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    warmup_steps: int = 500
    grad_accum_steps: int = 1
    clip_grad_norm: float = 1.0

    # === Data Configuration ===
    max_seq_len: int = 512
    vocab_size: int = 32000
    val_split: float = 0.1
    test_split: float = 0.1

    # === Field Configuration ===
    field_dim: int = 16
    spatial_size: tuple = (8, 8)
    adaptive_field: bool = True

    # === Optimization ===
    use_amp: bool = True
    patience: int = 3

    # === Logging & Checkpoints ===
    log_interval: int = 50
    eval_interval: int = 500
    save_interval: int = 1000
    output_dir: str = './checkpoints'

    # === Embeddings ===
    img_size: int = 224
    patch_size: int = 16
    n_frames: int = 8  # For video

    # === Loss Functions ===
    label_smoothing: float = 0.0
    contrastive_temperature: float = 0.07

    # === Inference ===
    inference_max_tokens: int = 100
    inference_temperature: float = 0.7

    # === System ===
    num_workers: int = 4
    seed: int = 42
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'

    def __post_init__(self):
        """Validate configuration and set derived values."""
        # Set default d_ff if not specified
        if self.d_ff is None:
            self.d_ff = 4 * self.d_model

        # Validate head divisibility
        if self.d_model % self.n_heads != 0:
            raise ValueError(
                f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})"
            )

        # Validate field dimension
        if self.field_dim < 16:
            import warnings
            warnings.warn(
                f"field_dim={self.field_dim} < 16 may have insufficient DOF. "
                "See paper Implementation Note 1."
            )

        # Ensure spatial_size is tuple
        if isinstance(self.spatial_size, list):
            self.spatial_size = tuple(self.spatial_size)

    def to_dict(self) -> dict:
        """Convert config to dictionary for compatibility with existing code."""
        return {
            'd_model': self.d_model,
            'n_heads': self.n_heads,
            'n_layers': self.n_layers,
            'd_ff': self.d_ff,
            'dropout': self.dropout,
            'use_mamba': self.use_mamba,
            'n_mamba_layers': self.n_mamba_layers,
            'n_attention_layers': self.n_attention_layers,
            'batch_size': self.batch_size,
            'max_epochs': self.max_epochs,
            'lr': self.learning_rate,
            'weight_decay': self.weight_decay,
            'warmup_steps': self.warmup_steps,
            'grad_accum_steps': self.grad_accum_steps,
            'clip_grad_norm': self.clip_grad_norm,
            'max_seq_len': self.max_seq_len,
            'vocab_size': self.vocab_size,
            'val_split': self.val_split,
            'test_split': self.test_split,
            'field_dim': self.field_dim,
            'spatial_size': list(self.spatial_size),
            'adaptive_field': self.adaptive_field,
            'use_amp': self.use_amp,
            'patience': self.patience,
            'log_interval': self.log_interval,
            'eval_interval': self.eval_interval,
            'save_interval': self.save_interval,
            'output_dir': self.output_dir,
            'num_workers': self.num_workers,
            'seed': self.seed,
            'device': self.device,
        }

    @classmethod
    def from_dict(cls, config_dict: dict) -> 'TrainingConfig':
        """Create config from dictionary."""
        # Map 'lr' to 'learning_rate' for compatibility
        if 'lr' in config_dict and 'learning_rate' not in config_dict:
            config_dict['learning_rate'] = config_dict.pop('lr')
        # Filter to only valid fields
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in config_dict.items() if k in valid_fields}
        return cls(**filtered)


# =============================================================================
# Training Presets
# =============================================================================

# Small model - fast training, good for testing
TRAINING_SMALL = TrainingConfig(
    d_model=256,
    n_heads=4,
    n_layers=4,
    n_mamba_layers=4,
    n_attention_layers=0,
    batch_size=128,
    max_epochs=10,
    learning_rate=1e-4,
    max_seq_len=512,
    output_dir='./checkpoints/small'
)

# Medium model - balanced performance
TRAINING_MEDIUM = TrainingConfig(
    d_model=512,
    n_heads=8,
    n_layers=6,
    n_mamba_layers=4,
    n_attention_layers=2,
    batch_size=64,
    max_epochs=10,
    learning_rate=3e-4,
    max_seq_len=512,
    output_dir='./checkpoints/medium'
)

# Large model - maximum capability
TRAINING_LARGE = TrainingConfig(
    d_model=768,
    n_heads=12,
    n_layers=12,
    n_mamba_layers=8,
    n_attention_layers=4,
    batch_size=32,
    max_epochs=10,
    learning_rate=3e-4,
    max_seq_len=512,
    output_dir='./checkpoints/large'
)

# Geometric quick training - optimized for demonstrating O(N) complexity
TRAINING_GEOMETRIC = TrainingConfig(
    d_model=256,
    n_heads=4,
    n_layers=4,
    n_mamba_layers=4,
    n_attention_layers=0,  # Pure Mamba = O(N)
    batch_size=128,
    max_epochs=5,
    learning_rate=1e-4,
    max_seq_len=1024,  # Longer sequences to show O(N) benefit
    log_interval=10,
    output_dir='./checkpoints/geometric'
)

# Test configuration - minimal settings for unit tests
TRAINING_TEST = TrainingConfig(
    d_model=128,
    n_heads=4,
    n_layers=2,
    n_mamba_layers=2,
    n_attention_layers=0,
    batch_size=4,
    max_epochs=1,
    learning_rate=1e-3,
    max_seq_len=64,
    vocab_size=256,
    field_dim=16,  # Keep >= 16 for DOF requirements
    spatial_size=(8, 8),
    output_dir='./checkpoints/test'
)


def get_training_config(size: str = 'medium') -> TrainingConfig:
    """
    Get a preset training configuration by size.

    Args:
        size: One of 'small', 'medium', 'large', 'geometric', 'test'

    Returns:
        TrainingConfig preset
    """
    presets = {
        'small': TRAINING_SMALL,
        'medium': TRAINING_MEDIUM,
        'large': TRAINING_LARGE,
        'geometric': TRAINING_GEOMETRIC,
        'test': TRAINING_TEST,
    }

    if size not in presets:
        raise ValueError(f"Unknown preset '{size}'. Choose from: {list(presets.keys())}")

    return presets[size]

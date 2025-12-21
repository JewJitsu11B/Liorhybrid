"""
Field State Extraction

Converts the cognitive tensor field T_ij(x,t) into Key and Value vectors
for geometric attention.

The field is the "memory" - its current state encodes all past evolution.
Keys and Values are projections of this state into the attention mechanism.

Architecture:
- Each spatial location (x,y) becomes a "memory token"
- T_ij provides rich 16D state (4×4 complex = 32 real DOF)
- Keys capture "what is in memory"
- Values capture "what to retrieve from memory"
"""

import torch
import torch.nn as nn


def flatten_field_state(T_field: torch.Tensor) -> torch.Tensor:
    """
    Flatten T_ij field into sequential memory tokens.

    Args:
        T_field: Cognitive tensor field (N_x, N_y, D, D) complex

    Returns:
        Flattened field (N_x * N_y, 2*D*D) real

    Each spatial location becomes a token with 2*D*D features
    (real and imaginary parts of D×D tensor).
    """
    N_x, N_y, D, D_out = T_field.shape
    assert D == D_out, "T_field must be square"

    # Convert complex to real representation
    # Stack real and imaginary parts: [real_00, imag_00, real_01, imag_01, ...]
    T_real = T_field.real  # (N_x, N_y, D, D)
    T_imag = T_field.imag  # (N_x, N_y, D, D)

    # Interleave real and imaginary: (N_x, N_y, D, D, 2)
    T_complex_stacked = torch.stack([T_real, T_imag], dim=-1)

    # Flatten spatial dimensions: (N_x * N_y, D, D, 2)
    T_flat = T_complex_stacked.reshape(N_x * N_y, D, D, 2)

    # Flatten tensor dimensions: (N_x * N_y, D * D * 2)
    T_tokens = T_flat.reshape(N_x * N_y, D * D * 2)

    return T_tokens


class FieldToKeyValue(nn.Module):
    """
    Learnable projection from T_ij field state to Key and Value vectors.

    This is the bridge from physics (field evolution) to cognition (attention).

    Architecture:
    - Input: T_ij field (N_x, N_y, D, D) complex
    - Flatten to (N_x * N_y, 2*D*D) tokens
    - Linear projection to (N_tokens, d_model) for K and V
    - Optional: Add positional encoding based on (x,y) coordinates
    - Optional: Add temporal encoding based on field history
    """

    def __init__(
        self,
        field_dim: int,  # D (dimension of T_ij tensor)
        d_model: int,    # Transformer hidden dimension
        use_positional_encoding: bool = True,
        use_temporal_encoding: bool = False
    ):
        """
        Initialize field-to-KV projection.

        Args:
            field_dim: Dimension D of T_ij tensor (e.g., 4)
            d_model: Target dimension for K, V (e.g., 512)
            use_positional_encoding: Add spatial (x,y) position info
            use_temporal_encoding: Add temporal evolution info
        """
        super().__init__()

        self.field_dim = field_dim
        self.d_model = d_model
        self.use_pos_enc = use_positional_encoding
        self.use_temp_enc = use_temporal_encoding

        # Input dimension: 2*D*D (real + imag for D×D tensor)
        input_dim = 2 * field_dim * field_dim

        # Learnable projections
        self.key_projection = nn.Linear(input_dim, d_model)
        self.value_projection = nn.Linear(input_dim, d_model)

        # Optional positional encoding (sinusoidal)
        if use_positional_encoding:
            # Will be added at forward time based on grid coordinates
            self.register_buffer('pos_scale', torch.tensor(1.0))

        # Optional temporal encoding
        if use_temporal_encoding:
            self.temporal_projection = nn.Linear(1, d_model)

    def forward(
        self,
        T_field: torch.Tensor,
        time: float = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Extract Keys and Values from field state.

        Args:
            T_field: Cognitive tensor field (N_x, N_y, D, D) complex
            time: Current time t for temporal encoding (optional)

        Returns:
            Keys: (N_tokens, d_model) where N_tokens = N_x * N_y
            Values: (N_tokens, d_model)

        Each spatial location in the field becomes one memory token.
        """
        N_x, N_y, D, D_out = T_field.shape
        device = T_field.device

        # Flatten field to tokens
        T_tokens = flatten_field_state(T_field)  # (N_x * N_y, 2*D*D)

        # Cast to projection layer dtype for FP16 training
        target_dtype = self.key_projection.weight.dtype
        T_tokens = T_tokens.to(target_dtype)

        # Project to K, V
        K = self.key_projection(T_tokens)      # (N_tokens, d_model)
        V = self.value_projection(T_tokens)    # (N_tokens, d_model)

        # Add positional encoding (cast to match K/V dtype)
        if self.use_pos_enc:
            pos_enc = self._create_positional_encoding(N_x, N_y, device).to(K.dtype)
            K = K + pos_enc
            V = V + pos_enc

        # Add temporal encoding (cast to match K/V dtype)
        if self.use_temp_enc and time is not None:
            time_tensor = torch.tensor([[time]], device=device, dtype=K.dtype)
            time_enc = self.temporal_projection(time_tensor)  # (1, d_model)
            K = K + time_enc
            V = V + time_enc

        return K, V

    def _create_positional_encoding(
        self,
        N_x: int,
        N_y: int,
        device: torch.device
    ) -> torch.Tensor:
        """
        Create sinusoidal positional encoding for 2D spatial grid.

        Uses separate encodings for x and y coordinates, similar to
        2D positional encoding in vision transformers.

        Args:
            N_x, N_y: Spatial grid dimensions
            device: Device for tensor creation

        Returns:
            Positional encoding (N_x * N_y, d_model)
        """
        # Create coordinate grids
        x_coords = torch.arange(N_x, device=device, dtype=torch.float32)
        y_coords = torch.arange(N_y, device=device, dtype=torch.float32)

        # Normalize to [0, 1]
        x_coords = x_coords / N_x
        y_coords = y_coords / N_y

        # Create meshgrid
        x_grid, y_grid = torch.meshgrid(x_coords, y_coords, indexing='ij')

        # Flatten to sequence
        x_flat = x_grid.reshape(-1, 1)  # (N_tokens, 1)
        y_flat = y_grid.reshape(-1, 1)  # (N_tokens, 1)

        # Sinusoidal encoding
        # PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
        # PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

        half_d = self.d_model // 2
        div_term = torch.exp(
            torch.arange(0, half_d, dtype=torch.float32, device=device) *
            -(torch.log(torch.tensor(10000.0)) / half_d)
        )

        # X encoding (first half of dimensions)
        pe_x = torch.zeros(N_x * N_y, half_d, device=device)
        pe_x[:, 0::2] = torch.sin(x_flat * div_term[0::2])
        pe_x[:, 1::2] = torch.cos(x_flat * div_term[1::2])

        # Y encoding (second half of dimensions)
        pe_y = torch.zeros(N_x * N_y, half_d, device=device)
        pe_y[:, 0::2] = torch.sin(y_flat * div_term[0::2])
        pe_y[:, 1::2] = torch.cos(y_flat * div_term[1::2])

        # Concatenate x and y encodings
        pos_enc = torch.cat([pe_x, pe_y], dim=1)  # (N_tokens, d_model)

        # Handle odd d_model
        if self.d_model % 2 == 1:
            pos_enc = torch.cat([pos_enc, torch.zeros(N_x * N_y, 1, device=device)], dim=1)

        return pos_enc


def extract_keys_values_from_field(
    T_field: torch.Tensor,
    field_to_kv: FieldToKeyValue,
    time: float = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Convenience function to extract K, V from field.

    Args:
        T_field: Cognitive tensor field (N_x, N_y, D, D) complex
        field_to_kv: Trained FieldToKeyValue projection module
        time: Optional time for temporal encoding

    Returns:
        Keys: (N_tokens, d_model)
        Values: (N_tokens, d_model)

    This function wraps the FieldToKeyValue module for easy use.
    """
    return field_to_kv(T_field, time=time)

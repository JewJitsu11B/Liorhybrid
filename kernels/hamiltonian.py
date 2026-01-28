"""
Hamiltonian Evolution Operator

Implements H[T] = -(ℏ²/2m)∇²T + V·T with anisotropic metric support.

TETRAD (VIELBEIN) CONNECTION:

The tetrad e^a_μ connects the metric g_μν to the Clifford algebra basis γ_a:
    
    g_μν = η_ab e^a_μ e^b_ν
    
For diagonal metrics: e^a_μ = diag(√g^xx, √g^yy)

This allows the Laplace-Beltrami operator to be expressed via the tetrad:
    ∇²_g = e^a_μ e^a_ν ∂²/∂x^μ∂x^ν

The tetrad also connects to the Clifford-Chevalley connection in causal_field.py:
    Γ_μ = e^a_μ γ_a
    
See: kernels/tetrad.py for full implementation

USAGE WITH CONTEXT MANAGER (RECOMMENDED):

    from kernels.hamiltonian import hamiltonian_evolution_with_metric
    from kernels.metric_context import MetricContext
    from kernels.tetrad import Tetrad
    
    # Create tetrad for metric-Clifford connection
    tetrad = Tetrad(dim=2)
    g_inv_diag = torch.tensor([10.0, 1.0, 1.0, ...])
    e = tetrad.compute_from_metric(g_inv_diag)
    
    # Context manager provides automatic validation & performance tracking
    with MetricContext(g_inv_diag, validate=True, track_perf=True) as ctx:
        H_T = hamiltonian_evolution_with_metric(
            T, hbar_cog=0.1, m_cog=1.0, g_inv_diag=ctx.g_inv
        )
        print(f"Tetrad:\n{e}")
        print(f"Is isotropic: {ctx.is_isotropic}")
        print(f"Time taken: {ctx.elapsed_time:.3f}s")

WHAT THE CONTEXT MANAGER DOES:
- ✅ Validates metric is positive definite (catches errors early)
- ✅ Tracks GPU time with proper synchronization  
- ✅ Provides helpful utilities (is_isotropic, get_spatial_components)
- ✅ Exception-safe cleanup (handles errors gracefully)
- ✅ Zero overhead when not tracking performance

Paper References:
- Equation (2): Hamiltonian definition
- Implementation Note 2: Discrete Laplacian
- models/causal_field.py: Clifford connection with tetrad
"""

import torch
import torch.nn.functional as F


def spatial_laplacian(T: torch.Tensor, dx: float = 1.0) -> torch.Tensor:
    """
    Compute spatial Laplacian ∇²T via finite differences.

    Paper Equation (2), discrete form:
        ∇²T_ij[x,y] = (T[x±1,y] + T[x,y±1] - 4T[x,y]) / dx²

    Args:
        T: Tensor field of shape (N_x, N_y, D, D)
        dx: Grid spacing (default 1.0)

    Returns:
        Laplacian of same shape as T

    Implementation Note 2 from paper:
        Uses 2D convolution with [[0,1,0],[1,-4,1],[0,1,0]] kernel
        with PERIODIC boundary conditions (circular padding)
    """
    N_x, N_y, D, D_out = T.shape

    # Reshape for 2D convolution: (1, D*D, N_x, N_y)
    T_reshaped = T.permute(2, 3, 0, 1).reshape(1, D*D_out, N_x, N_y)

    # Laplacian kernel (paper Implementation Note 2)
    kernel = torch.tensor(
        [[0, 1, 0],
         [1, -4, 1],
         [0, 1, 0]],
        dtype=T.dtype,
        device=T.device
    ).reshape(1, 1, 3, 3) / dx**2

    # Expand kernel for all channels
    kernel = kernel.repeat(D*D_out, 1, 1, 1)

    # BUG FIX: Apply periodic boundary conditions (circular padding)
    # This ensures proper physics at boundaries
    T_padded = F.pad(T_reshaped, (1, 1, 1, 1), mode='circular')
    laplacian = F.conv2d(
        T_padded,
        kernel,
        padding=0,  # No padding needed - already done with circular
        groups=D*D_out
    )

    # Reshape back: (N_x, N_y, D, D)
    laplacian = laplacian.reshape(D, D_out, N_x, N_y).permute(2, 3, 0, 1)

    return laplacian


def spatial_laplacian_x(T: torch.Tensor, dx: float = 1.0) -> torch.Tensor:
    """
    Compute second derivative in x-direction: ∂²T/∂x²
    
    Uses finite differences: (T[x+1,y] - 2T[x,y] + T[x-1,y]) / dx²
    with PERIODIC boundary conditions.
    
    Args:
        T: Tensor field of shape (N_x, N_y, D, D)
        dx: Grid spacing (default 1.0)
    
    Returns:
        Second derivative in x (same shape as T)
    """
    N_x, N_y, D, D_out = T.shape
    
    # Reshape for 2D convolution: (1, D*D, N_x, N_y)
    T_reshaped = T.permute(2, 3, 0, 1).reshape(1, D*D_out, N_x, N_y)
    
    # Second derivative kernel in x-direction: [1, -2, 1] horizontally
    kernel = torch.tensor(
        [[0, 0, 0],
         [1, -2, 1],
         [0, 0, 0]],
        dtype=T.dtype,
        device=T.device
    ).reshape(1, 1, 3, 3) / dx**2
    
    # Expand kernel for all channels
    kernel = kernel.repeat(D*D_out, 1, 1, 1)
    
    # BUG FIX: Apply periodic boundary conditions
    T_padded = F.pad(T_reshaped, (1, 1, 1, 1), mode='circular')
    d2_dx2 = F.conv2d(
        T_padded,
        kernel,
        padding=0,
        groups=D*D_out
    )
    
    # Reshape back: (N_x, N_y, D, D)
    d2_dx2 = d2_dx2.reshape(D, D_out, N_x, N_y).permute(2, 3, 0, 1)
    
    return d2_dx2


def spatial_laplacian_y(T: torch.Tensor, dy: float = 1.0) -> torch.Tensor:
    """
    Compute second derivative in y-direction: ∂²T/∂y²
    
    Uses finite differences: (T[x,y+1] - 2T[x,y] + T[x,y-1]) / dy²
    with PERIODIC boundary conditions.
    
    Args:
        T: Tensor field of shape (N_x, N_y, D, D)
        dy: Grid spacing (default 1.0)
    
    Returns:
        Second derivative in y (same shape as T)
    """
    N_x, N_y, D, D_out = T.shape
    
    # Reshape for 2D convolution: (1, D*D, N_x, N_y)
    T_reshaped = T.permute(2, 3, 0, 1).reshape(1, D*D_out, N_x, N_y)
    
    # Second derivative kernel in y-direction: [1, -2, 1] vertically
    kernel = torch.tensor(
        [[0, 1, 0],
         [0, -2, 0],
         [0, 1, 0]],
        dtype=T.dtype,
        device=T.device
    ).reshape(1, 1, 3, 3) / dy**2
    
    # Expand kernel for all channels
    kernel = kernel.repeat(D*D_out, 1, 1, 1)
    
    # BUG FIX: Apply periodic boundary conditions
    T_padded = F.pad(T_reshaped, (1, 1, 1, 1), mode='circular')
    d2_dy2 = F.conv2d(
        T_padded,
        kernel,
        padding=0,
        groups=D*D_out
    )
    
    # Reshape back: (N_x, N_y, D, D)
    d2_dy2 = d2_dy2.reshape(D, D_out, N_x, N_y).permute(2, 3, 0, 1)
    
    return d2_dy2


def hamiltonian_evolution(
    T: torch.Tensor,
    hbar_cog: float = 0.1,
    m_cog: float = 1.0,
    V: torch.Tensor = None
) -> torch.Tensor:
    """
    Compute H[T] = -(ℏ²/2m)∇²T + V·T

    Paper Equation (2)

    Args:
        T: Tensor field (N_x, N_y, D, D)
        hbar_cog: ℏ_cog cognitive Planck constant
        m_cog: m_cog effective mass
        V: Optional potential V_ij(x,y) (same shape as T)

    Returns:
        H[T] of same shape

    Note: For common potential landscapes, use create_potential() helper.
    """
    # Kinetic term: -(ℏ²/2m)∇²T
    lap_T = spatial_laplacian(T, dx=1.0)
    kinetic = -(hbar_cog**2 / (2 * m_cog)) * lap_T

    # Potential term: V·T
    if V is not None:
        potential = V * T
    else:
        potential = 0.0

    return kinetic + potential


def hamiltonian_evolution_with_metric(
    T: torch.Tensor,
    hbar_cog: float,
    m_cog: float,
    g_inv_diag: torch.Tensor = None,
    V: torch.Tensor = None
) -> torch.Tensor:
    """
    Metric-aware Hamiltonian: H[T] = -(ℏ²/2m)∇²_g T + V·T
    
    For diagonal spatial metric g_ij = diag(g_xx, g_yy):
        ∇²_g T = g^xx ∂²T/∂x² + g^yy ∂²T/∂y²
    
    This ensures field evolution respects the learned Riemannian geometry
    instead of assuming flat (Euclidean) space.
    
    IMPORTANT LIMITATIONS (for 100% mathematical accuracy):
    1. **Spatially constant metrics only**: This implementation assumes
       g^xx and g^yy are CONSTANT across space. For spatially varying
       metrics g^ij(x,y), you need Christoffel symbols:
       ∇²_g f = g^ij ∂²f/∂x^i∂x^j + (∂_i g^ij) ∂f/∂x^j
       
    2. **Diagonal metrics only**: For off-diagonal metrics, need cross terms:
       ∇²_g f = g^xx ∂²f/∂x² + g^yy ∂²f/∂y² + 2g^xy ∂²f/∂x∂y
       
    3. **Rank-2 tensor field**: T is (N_x, N_y, D, D) - a rank-2 tensor
       at each spatial point. Current implementation treats the (D,D)
       bundle with diagonal internal metric. For full covariant derivative
       on tensor bundles, need connection coefficients in internal space.
    
    Valid for: Spatially constant diagonal metrics on scalar-like fields.
    Use with: Learned metrics from trainer2.py (constant per epoch).
    
    Args:
        T: Tensor field (N_x, N_y, D, D) complex
        hbar_cog: Cognitive Planck constant ℏ_cog
        m_cog: Effective mass m_cog
        g_inv_diag: Inverse metric diagonal (n,) where n >= 2
                    First two components used for spatial directions (x, y)
                    Must be POSITIVE (checked) and CONSTANT (assumed)
                    If None, falls back to flat-space evolution
        V: Optional potential V(x,y) of same shape as T
    
    Returns:
        H[T]: Hamiltonian evolution term (same shape as T)
    
    Physics:
        Standard Hamiltonian assumes Euclidean metric (dx² + dy²).
        In curved space with metric g_ij, the Laplacian becomes:
            ∇²_g = (1/√g) ∂_i(√g g^ij ∂_j)
        
        For DIAGONAL metric in 2D spatial coordinates:
            ∇²_g T = g^xx ∂²T/∂x² + g^yy ∂²T/∂y²
        
        This formula is exact for spatially constant diagonal metrics.
    
    Current Implementation: ANISOTROPIC SCALING
        - Uses g^xx for x-direction: g^xx * ∂²T/∂x²
        - Uses g^yy for y-direction: g^yy * ∂²T/∂y²
        - Respects directional differences in geometry
        - Reduces to isotropic when g^xx = g^yy
        - Periodic boundary conditions (physically correct for fields)
        - GPU-optimized (no CPU synchronization)
        - Validates positive definiteness automatically
    
    References:
        - Laplace-Beltrami operator on Riemannian manifolds
        - models/causal_field.py (Clifford-Chevalley connections)
        - docs/Clifford_hodge_Chevally.pdf
    """
    if g_inv_diag is None:
        # No metric provided: use flat-space (Euclidean)
        return hamiltonian_evolution(T, hbar_cog, m_cog, V)
    
    # === ANISOTROPIC METRIC SCALING ===
    # Compute directional second derivatives
    d2_dx2 = spatial_laplacian_x(T, dx=1.0)  # ∂²T/∂x²
    d2_dy2 = spatial_laplacian_y(T, dy=1.0)  # ∂²T/∂y²
    
    # Extract metric components for spatial directions (x, y)
    # BUG FIX: Keep on GPU (don't call .item() for performance)
    if g_inv_diag.dim() == 1 and g_inv_diag.shape[0] >= 2:
        # Validate metric is positive definite (required for Riemannian geometry)
        if torch.any(g_inv_diag[:2] <= 0):
            raise ValueError(
                f"Metric must be positive definite for Riemannian geometry. "
                f"Got g_xx={g_inv_diag[0]:.4f}, g_yy={g_inv_diag[1]:.4f}"
            )
        if torch.any(~torch.isfinite(g_inv_diag[:2])):
            raise ValueError(f"Metric contains NaN or Inf: {g_inv_diag[:2]}")
        
        # Use first two components for x and y directions (keep on GPU)
        g_xx = g_inv_diag[0]  # Tensor, not scalar - stays on GPU
        g_yy = g_inv_diag[1]  # Tensor, not scalar - stays on GPU
    elif g_inv_diag.dim() == 1 and g_inv_diag.shape[0] == 1:
        # Only one component: use isotropically
        if g_inv_diag[0] <= 0:
            raise ValueError(f"Metric must be positive: got {g_inv_diag[0]:.4f}")
        g_xx = g_yy = g_inv_diag[0]
    else:
        # Fallback: use flat space
        g_xx = g_yy = 1.0
    
    # Anisotropic Laplacian: ∇²_g T = g^xx ∂²T/∂x² + g^yy ∂²T/∂y²
    lap_T_aniso = g_xx * d2_dx2 + g_yy * d2_dy2
    
    # Kinetic term (metric-aware)
    kinetic = -(hbar_cog**2 / (2 * m_cog)) * lap_T_aniso
    
    # Potential term (unchanged by metric)
    potential = V * T if V is not None else 0.0
    
    return kinetic + potential


def create_potential(
    spatial_size: tuple,
    tensor_dim: int,
    potential_type: str = "harmonic",
    strength: float = 1.0,
    center: tuple = None,
    device: str = 'cpu',
    dtype: torch.dtype = torch.complex64
) -> torch.Tensor:
    """
    Create common potential landscapes V(x,y) for the Hamiltonian.

    Args:
        spatial_size: (N_x, N_y) grid dimensions
        tensor_dim: D (tensor dimension at each point)
        potential_type: Type of potential:
            - "harmonic": V(x,y) = ½k(x² + y²) (oscillator)
            - "gaussian_well": V(x,y) = -A exp(-(x² + y²)/2σ²)
            - "gaussian_barrier": V(x,y) = +A exp(-(x² + y²)/2σ²)
            - "constant": V(x,y) = constant (shift energy baseline)
            - "zero": V(x,y) = 0 (free field)
        strength: Potential strength parameter (k for harmonic, A for Gaussian)
        center: (x0, y0) center point (default: grid center)
        device: torch device
        dtype: tensor dtype

    Returns:
        Potential tensor V of shape (N_x, N_y, D, D)

    Physical Interpretation:
        - Harmonic: Confines field to center (attractive)
        - Gaussian well: Local attractive region
        - Gaussian barrier: Local repulsive region
        - Constant: Energy offset (no force)

    Example:
        >>> V = create_potential((28, 28), 16, "harmonic", strength=0.1)
        >>> H_T = hamiltonian_evolution(T, V=V)
    """
    N_x, N_y = spatial_size
    D = tensor_dim

    # Default center to grid center
    if center is None:
        center = (N_x / 2.0, N_y / 2.0)

    x0, y0 = center

    # Create spatial coordinate grids
    x = torch.arange(N_x, dtype=torch.float32, device=device)
    y = torch.arange(N_y, dtype=torch.float32, device=device)
    X, Y = torch.meshgrid(x, y, indexing='ij')

    # Compute potential at each spatial point
    if potential_type == "harmonic":
        # V(x,y) = ½k((x-x0)² + (y-y0)²)
        r_squared = (X - x0)**2 + (Y - y0)**2
        V_spatial = 0.5 * strength * r_squared

    elif potential_type == "gaussian_well":
        # V(x,y) = -A exp(-r²/2σ²), σ = N/4 for reasonable width
        sigma = min(N_x, N_y) / 4.0
        r_squared = (X - x0)**2 + (Y - y0)**2
        V_spatial = -strength * torch.exp(-r_squared / (2 * sigma**2))

    elif potential_type == "gaussian_barrier":
        # V(x,y) = +A exp(-r²/2σ²)
        sigma = min(N_x, N_y) / 4.0
        r_squared = (X - x0)**2 + (Y - y0)**2
        V_spatial = strength * torch.exp(-r_squared / (2 * sigma**2))

    elif potential_type == "constant":
        # V(x,y) = constant
        V_spatial = strength * torch.ones((N_x, N_y), device=device)

    elif potential_type == "zero":
        # V(x,y) = 0
        V_spatial = torch.zeros((N_x, N_y), device=device)

    else:
        raise ValueError(
            f"Unknown potential_type '{potential_type}'. "
            f"Choose from: harmonic, gaussian_well, gaussian_barrier, constant, zero"
        )

    # Expand to full tensor shape (N_x, N_y, D, D)
    # Potential acts as V·I on the D×D tensor at each point
    V = torch.zeros((N_x, N_y, D, D), dtype=dtype, device=device)

    for i in range(D):
        V[:, :, i, i] = V_spatial  # Diagonal elements

    return V

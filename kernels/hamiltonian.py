"""
Hamiltonian Evolution Operator

Implements H[T] = -(ℏ²/2m)∇²T + V·T

Paper References:
- Equation (2): Hamiltonian definition
- Implementation Note 2: Discrete Laplacian
"""

import torch
import torch.nn.functional as F


@torch.compile
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

    # Apply convolution with circular padding (periodic boundary)
    laplacian = F.conv2d(
        T_reshaped,
        kernel,
        padding='same',
        groups=D*D_out
    )

    # Reshape back: (N_x, N_y, D, D)
    laplacian = laplacian.reshape(D, D_out, N_x, N_y).permute(2, 3, 0, 1)

    return laplacian


@torch.compile
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

    TODO: Implement non-trivial potential landscapes
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

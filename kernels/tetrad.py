"""
Tetrad (Vielbein) Field for Metric-Clifford Connection

A tetrad e^a_μ is an orthonormal frame field that connects:
1. The curved coordinate basis {∂/∂x^μ} 
2. The flat Clifford algebra basis {γ^a}

Key Relations:
    g_μν = η_ab e^a_μ e^b_ν    (metric in terms of tetrad)
    e^a_μ e^μ_b = δ^a_b        (orthonormality)
    
For diagonal metrics g_μν = diag(g_xx, g_yy), the tetrad is:
    e^a_μ = diag(√g_xx, √g_yy, ...)

This connects the anisotropic metric to the Clifford-Chevalley 
connection used in causal_field.py.

References:
    - models/causal_field.py (Gamma connection with tetrad)
    - docs/Clifford_hodge_Chevally.pdf
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple


class Tetrad(nn.Module):
    """
    Tetrad (vielbein) field e^a_μ connecting metric to Clifford algebra.
    
    For a Riemannian metric g_μν, the tetrad satisfies:
        g_μν = η_ab e^a_μ e^b_ν
    
    where η_ab is the flat Minkowski/Euclidean metric.
    
    Usage:
        tetrad = Tetrad(dim=2, learnable=False)
        e = tetrad.compute_from_metric(g_inv_diag)
        # e is shape (dim, dim) - the vielbein matrix
    """
    
    def __init__(self, dim: int = 2, learnable: bool = False, device='cpu'):
        """
        Initialize tetrad field.
        
        Args:
            dim: Spatial dimension (2 for x,y; 4 for spacetime)
            learnable: Whether tetrad is learnable parameter (default: compute from metric)
            device: Device to place tensors on
        """
        super().__init__()
        self.dim = dim
        self.device = device
        
        if learnable:
            # Learnable tetrad (for training)
            self.e = nn.Parameter(torch.eye(dim, device=device))
        else:
            # Computed from metric (default)
            self.register_buffer('e', torch.eye(dim, device=device))
    
    def compute_from_metric(
        self, 
        g_inv_diag: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute tetrad from diagonal inverse metric g^μν.
        
        For diagonal metric: e^a_μ = diag(√g^{xx}, √g^{yy})
        
        Args:
            g_inv_diag: Inverse metric diagonal (n,) where n >= dim
        
        Returns:
            e: Tetrad field (dim, dim) matrix
        """
        # Extract spatial components
        if g_inv_diag.shape[0] >= self.dim:
            g_components = g_inv_diag[:self.dim]
        else:
            # Pad with ones if needed
            g_components = torch.ones(self.dim, device=g_inv_diag.device)
            g_components[:g_inv_diag.shape[0]] = g_inv_diag
        
        # For diagonal metric, tetrad is diagonal with √g components
        # Note: g_inv_diag contains g^μν, so tetrad element is √g^μν
        e = torch.diag(torch.sqrt(g_components))
        
        return e
    
    def compute_inverse(self, e: torch.Tensor) -> torch.Tensor:
        """
        Compute inverse tetrad e_μ^a (upper index).
        
        For diagonal tetrad: inverse is also diagonal with 1/√g components.
        
        Args:
            e: Tetrad (dim, dim)
        
        Returns:
            e_inv: Inverse tetrad (dim, dim)
        """
        # For diagonal tetrad, inverse is just reciprocal of diagonal
        if torch.allclose(e, torch.diag(torch.diag(e)), atol=1e-6):
            # Diagonal case (fast)
            diag_inv = 1.0 / torch.diag(e)
            return torch.diag(diag_inv)
        else:
            # General case (slower)
            return torch.linalg.inv(e)
    
    def verify_orthonormality(
        self, 
        e: torch.Tensor,
        atol: float = 1e-5
    ) -> Tuple[bool, float]:
        """
        Verify e^a_μ e^μ_b = δ^a_b (orthonormality condition).
        
        Args:
            e: Tetrad (dim, dim)
            atol: Absolute tolerance
        
        Returns:
            (is_orthonormal, max_error)
        """
        e_inv = self.compute_inverse(e)
        product = e @ e_inv
        identity = torch.eye(self.dim, device=e.device, dtype=e.dtype)
        
        error = torch.abs(product - identity)
        max_error = torch.max(error).item()
        
        return max_error < atol, max_error
    
    def contract_with_clifford(
        self,
        e: torch.Tensor,
        gamma_matrices: torch.Tensor
    ) -> torch.Tensor:
        """
        Contract tetrad with Clifford gamma matrices to get connection.
        
        Gamma_μ = e^a_μ γ_a
        
        This is what models/causal_field.py does to get the Clifford connection.
        
        Args:
            e: Tetrad (dim, dim)
            gamma_matrices: Clifford matrices (dim, d, d)
        
        Returns:
            Gamma: Contracted connection (dim, d, d)
        """
        # Contract: Gamma_μ = sum_a e^a_μ γ_a
        # Using einsum: 'am,aij->mij'
        Gamma = torch.einsum('am,aij->mij', e, gamma_matrices)
        return Gamma
    
    def forward(
        self,
        g_inv_diag: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute or return tetrad.
        
        Args:
            g_inv_diag: If provided, compute tetrad from metric
        
        Returns:
            e: Tetrad field (dim, dim)
        """
        if g_inv_diag is not None:
            return self.compute_from_metric(g_inv_diag)
        else:
            return self.e


def compute_metric_from_tetrad(e: torch.Tensor) -> torch.Tensor:
    """
    Compute metric from tetrad: g_μν = η_ab e^a_μ e^b_ν
    
    For Euclidean signature: η_ab = δ_ab (Kronecker delta)
    For Minkowski signature: η_ab = diag(-1, 1, 1, 1)
    
    Args:
        e: Tetrad (dim, dim)
    
    Returns:
        g: Metric tensor (dim, dim)
    """
    # Euclidean case: g_μν = e^a_μ e^a_ν = (e^T @ e)
    g = e.T @ e
    return g


def anisotropic_laplacian_from_tetrad(
    T: torch.Tensor,
    e: torch.Tensor,
    spatial_laplacian_x_fn,
    spatial_laplacian_y_fn
) -> torch.Tensor:
    """
    Compute anisotropic Laplacian using tetrad.
    
    ∇²_g T = g^μν ∂²T/∂x^μ∂x^ν = (e^a_μ e^a_ν) ∂²T/∂x^μ∂x^ν
    
    For diagonal tetrad with 2D spatial:
        ∇²_g T = e^1_1² ∂²T/∂x² + e^2_2² ∂²T/∂y²
                = g^{xx} ∂²T/∂x² + g^{yy} ∂²T/∂y²
    
    Args:
        T: Field tensor
        e: Tetrad (2, 2)
        spatial_laplacian_x_fn: Function to compute ∂²T/∂x²
        spatial_laplacian_y_fn: Function to compute ∂²T/∂y²
    
    Returns:
        Anisotropic Laplacian
    """
    # Extract metric components from tetrad
    # For diagonal tetrad: e = diag(√g^xx, √g^yy)
    # So: g^xx = e[0,0]², g^yy = e[1,1]²
    g_xx = e[0, 0]**2
    g_yy = e[1, 1]**2
    
    # Compute directional derivatives
    d2_dx2 = spatial_laplacian_x_fn(T)
    d2_dy2 = spatial_laplacian_y_fn(T)
    
    # Anisotropic Laplacian
    lap_aniso = g_xx * d2_dx2 + g_yy * d2_dy2
    
    return lap_aniso


# Example usage
if __name__ == "__main__":
    print("Tetrad (Vielbein) Field Demo")
    print("=" * 50)
    
    # Create tetrad
    tetrad = Tetrad(dim=2)
    
    # Example: Anisotropic metric with g^xx=4, g^yy=1
    g_inv_diag = torch.tensor([4.0, 1.0])
    
    # Compute tetrad from metric
    e = tetrad.compute_from_metric(g_inv_diag)
    print(f"\nMetric diagonal: {g_inv_diag}")
    print(f"Tetrad:\n{e}")
    
    # Verify orthonormality
    is_ortho, error = tetrad.verify_orthonormality(e)
    print(f"\nOrthonormal: {is_ortho} (error: {error:.2e})")
    
    # Reconstruct metric
    g = compute_metric_from_tetrad(e)
    print(f"\nReconstructed metric:\n{g}")
    print(f"Expected diagonal: {g_inv_diag}")
    
    # Connection to Clifford algebra
    print("\n" + "=" * 50)
    print("Connection to Clifford Algebra")
    print("=" * 50)
    print("The tetrad e^a_μ connects:")
    print("1. Curved coords (x,y) with metric g_μν")
    print("2. Flat Clifford basis with gamma matrices γ_a")
    print("\nClifford connection: Γ_μ = e^a_μ γ_a")
    print("Used in: models/causal_field.py")

"""
Parallel Transport Operators for the Causal Accumulation Law.

Implements the transport structures required by the law:

    T^{mu nu}_{rho sigma}^a(x)
        = alpha  J^{mu nu}_{rho sigma}^a(x)
        - (1-alpha) integral_{J^-(x)} k(tau;x,x')
                Pi^{mu nu}_{rho sigma||alpha beta}^{eta upsilon}(x,x')
                P^alpha_{alpha'}(x,x')
                P^beta_{beta'}(x,x')
                Phi^a_b(x)
                P^b_c(x,x')
                J^{alpha' beta'}_{eta upsilon}^c(x')
                d^4x'

The two concrete transport structures are:

  * ``ParallelPropagator``    — P^alpha_{alpha'}(x, x')
        Propagates tangent-vector indices along a causal geodesic from
        x' to x.  Implemented as a learned linear map with an identity
        initialisation so that in flat spacetime it reduces to the
        trivial identity transport.

  * ``FiberHolonomy``          — Phi^a_b(x) composed with P^b_c(x, x')
        Combines the local bivector field Phi at x with a fiber-bundle
        parallel propagator that transports the internal index c at x'
        to the index b at x before applying Phi.

==========================================================================
Transport-Operator Implementation Plan
==========================================================================

Phase 1 – Flat-spacetime baseline (current)
--------------------------------------------
* ``ParallelPropagator``: learned d_field×d_field matrix initialised to
  the identity.  Provides the correct flat-limit P^α_{α'}=δ^α_{α'}.
* ``FiberHolonomy``: thin wrapper that composes the static bivector Phi
  with a learned fiber propagator.

Phase 2 – Geodesic connection (future)
---------------------------------------
* Derive P^α_{α'}(x,x') from the Christoffel symbols Γ of the
  CognitiveManifold using the O(1) LIoR recurrence for integration
  (see models/manifold.py geodesic_step).
* Expose curvature-corrected transport via the existing CognitiveManifold
  interface so that training can anneal from flat to curved as the
  manifold learns.

Phase 3 – Tail-corrected Green's function (future)
----------------------------------------------------
* Replace the scalar memory kernel k(τ) with the full bivector kernel
  k(τ;x,x') = w(τ) * G_del(x,x'), where G_del is the delayed Green's
  function of the relevant hyperbolic operator.
* In the GR regime G_del acquires tail terms (Hadamard parametrix);
  these can be added as a correction to the current LIoR convolution
  without breaking the O(1) recurrence.

Phase 4 – Fractional differential operator (future)
-----------------------------------------------------
* Wrap the entire accumulation law in the fractional derivative
  nabla^{(alpha)mu} to obtain the structural differential form:
      nabla^{(alpha)mu}[ alpha J - (1-alpha) integral ... ] = Phi^a_b J
  This requires extending the existing fractional_memory kernel
  (kernels/fractional_memory.py) to act on the full tensor T rather
  than on scalar time-series.
"""
try: import usage_tracker; usage_tracker.track(__file__)
except: pass

import math
import torch
import torch.nn as nn


class ParallelPropagator(nn.Module):
    """
    Parallel propagator P^alpha_{alpha'}(x, x').

    Transports one tangent-vector index from event x' to event x along
    the unique causal geodesic connecting them.

    In flat Minkowski spacetime this reduces to the identity:
        P^alpha_{alpha'} = delta^alpha_{alpha'}

    The implementation uses a learned linear map initialised to the
    identity so that the network can deform away from the flat case as
    it learns the underlying geometry.

    Args:
        d_field: Dimension of the field index space.
    """

    def __init__(self, d_field: int = 16):
        super().__init__()
        self.d_field = d_field
        # Learnable transport matrix; identity init = flat spacetime
        self.transport = nn.Parameter(torch.eye(d_field))

    def forward(self, v: torch.Tensor) -> torch.Tensor:
        """
        Transport a vector/tensor index from x' to x.

        Args:
            v: Tensor with last dimension ``d_field`` [..., d_field].

        Returns:
            Transported tensor of the same shape.
        """
        # P v  (last-axis contraction)
        return torch.einsum('...j,ij->...i', v, self.transport)

    def is_flat_limit(self, atol: float = 1e-6) -> bool:
        """Return True when the propagator is close to the identity."""
        return bool(
            torch.allclose(self.transport, torch.eye(self.d_field,
                                                      device=self.transport.device),
                           atol=atol)
        )


class FiberHolonomy(nn.Module):
    """
    Fiber holonomy Phi^a_b(x) composed with P^b_c(x, x').

    Combines the local bivector field Phi at event x with a fiber
    parallel-propagator P_fiber that transports the internal index from
    x' to x before the local action of Phi is applied:

        holonomy^a_c  =  Phi^a_b(x) * P^b_c(x, x')

    Args:
        d_internal: Dimension of the internal fiber (index a, b, c).
    """

    def __init__(self, d_internal: int = 16):
        super().__init__()
        self.d_internal = d_internal

        # Local bivector field Phi^a_b(x) — antisymmetric
        Phi_init = torch.randn(d_internal, d_internal) / math.sqrt(d_internal)
        self.Phi = nn.Parameter(Phi_init - Phi_init.T)

        # Fiber parallel propagator P^b_c(x, x') — identity init
        self.P_fiber = nn.Parameter(torch.eye(d_internal))

    @property
    def antisymmetric_phi(self) -> torch.Tensor:
        """Return the antisymmetrised Phi to enforce Phi^[ab] = -Phi^[ba]."""
        return 0.5 * (self.Phi - self.Phi.T)

    def forward(self, field: torch.Tensor) -> torch.Tensor:
        """
        Apply fiber holonomy to an internal-index tensor.

        Computes  output^a = Phi^a_b  P^b_c  field^c.

        Args:
            field: [..., d_internal] internal-index tensor at x'.

        Returns:
            [..., d_internal] holonomy-transformed tensor at x.
        """
        Phi = self.antisymmetric_phi  # [d, d]
        # First transport from x' to x: P_fiber @ field
        transported = torch.einsum('...c,bc->...b', field, self.P_fiber)
        # Then apply local Phi^a_b
        return torch.einsum('...b,ab->...a', transported, Phi)

    def holonomy_matrix(self) -> torch.Tensor:
        """
        Full d×d holonomy matrix:  H^a_c = Phi^a_b P^b_c.
        """
        return torch.einsum('ab,bc->ac', self.antisymmetric_phi, self.P_fiber)

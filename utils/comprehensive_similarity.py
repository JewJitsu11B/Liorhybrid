"""
Comprehensive 15D Similarity Vector Computation

Implements rich geometric similarity measures beyond simple cosine similarity.

IMPLEMENTATION PHASES:
- Phase 1: 9D core (cheap/medium measures) [THIS FILE]
- Phase 2: 12D extended (+ entropies) [FUTURE]
- Phase 3: 15D full (+ statistical, tiered) [FUTURE]

The 15D similarity vector captures comprehensive geometric relationships:
    1. cosine: Angular alignment
    2. wedge_magnitude: Rotational structure  
    3. tensor_trace: Interaction strength
    4. spinor_magnitude: Phase overlap
    5. spinor_phase: Phase angle
    6. energy: Field coupling
    7. l2_tangent: Euclidean in tangent space
    8. l1_tangent: Manhattan in tangent space
    9. lior_distance: Geodesic distance (PRIMARY)
    10. geodesic_kendall_tau: Non-parametric correlation [FUTURE]
    11. manifold_mutual_info: Information distance [FUTURE]
    12. variational_entropy_diff: Field entropy difference [FUTURE]
    13. renyi_entropy_diff: Collision entropy [FUTURE]
    14. local_curvature_diff: Curvature difference [FUTURE]
    15. sectional_curvature: 2-plane curvature [FUTURE]

References:
    - Problem Statement: Extend Neighbor Addressing with Comprehensive Similarity Scores
    - Computational Cost Analysis: ~400 FLOPs/candidate for 9D core
"""
try:
    import usage_tracker
    usage_tracker.track(__file__)
except:
    pass

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from ..models.manifold import CognitiveManifold


class ComprehensiveSimilarity:
    """
    Comprehensive similarity computation with tiered execution.
    
    Computes 15D similarity vectors capturing rich geometric relationships
    between query and candidate points. Uses tiered computation to balance
    accuracy and performance.
    
    Args:
        manifold: CognitiveManifold instance for geometric operations
        mode: Computation mode - 'core' (9D), 'extended' (12D), 'full' (15D)
        
    Attributes:
        manifold: Manifold for geodesic computations
        mode: Current computation mode
        context_cache: Cache for precomputed context statistics
    """
    
    def __init__(
        self,
        manifold: 'CognitiveManifold',
        mode: str = 'core'
    ):
        if mode not in ['core', 'extended', 'full']:
            raise ValueError(f"Invalid mode: {mode}. Must be 'core', 'extended', or 'full'")
        
        self.manifold = manifold
        self.mode = mode
        self.context_cache = {}
    
    @torch.inference_mode()
    def compute_batch(
        self,
        query_embedding: torch.Tensor,      # [d_embed]
        candidate_embeddings: torch.Tensor, # [N, d_embed]
        context: Optional[torch.Tensor] = None  # [M, d_embed]
    ) -> torch.Tensor:
        """
        Compute similarity vectors for all candidates (batched).
        
        Uses tiered computation:
        1. Compute cheap measures for all (9D core)
        2. Compute entropy measures for all if mode='extended' (12D)
        3. Compute expensive measures for top-K only if mode='full' (15D)
        
        Args:
            query_embedding: Query embedding [d_embed]
            candidate_embeddings: All candidate embeddings [N, d_embed]
            context: Context for statistical measures (optional) [M, d_embed]
        
        Returns:
            Similarity matrix [N, D] where D ∈ {9, 12, 15}
            
        Example:
            >>> manifold = CognitiveManifold(d_embed=512, d_coord=8)
            >>> sim = ComprehensiveSimilarity(manifold, mode='core')
            >>> query_emb = torch.randn(512)
            >>> candidates_emb = torch.randn(100, 512)
            >>> vectors = sim.compute_batch(query_emb, candidates_emb)  # [100, 9]
        """
        N = len(candidate_embeddings)
        
        # Project to coordinates for geometric operations
        query_coord = self.manifold.to_coords(query_embedding)  # [d_coord]
        candidate_coords = self.manifold.to_coords(candidate_embeddings)  # [N, d_coord]
        
        # Phase 1: Core 9D (all candidates)
        core_vectors = self._compute_core_batch(
            query_embedding, query_coord,
            candidate_embeddings, candidate_coords
        )  # [N, 9]
        
        if self.mode == 'core':
            return core_vectors
        
        # Phase 2: Extended 12D (all candidates, if requested)
        if self.mode in ['extended', 'full']:
            raise NotImplementedError(
                "Extended (12D) and Full (15D) modes not yet implemented. "
                "Use mode='core' for Phase 1 (9D core similarity)."
            )
        
        return core_vectors
    
    def _compute_core_batch(
        self,
        query_embedding: torch.Tensor,      # [d_embed]
        query_coord: torch.Tensor,          # [d_coord]
        candidate_embeddings: torch.Tensor, # [N, d_embed]
        candidate_coords: torch.Tensor      # [N, d_coord]
    ) -> torch.Tensor:
        """
        Compute 9D core similarity (cheap/medium measures).
        
        Returns: [N, 9] with dimensions:
            [cosine, wedge, tensor, spinor_mag, spinor_phase,
             energy, l2_tangent, l1_tangent, lior_distance]
             
        Cost: ~400 FLOPs per candidate (excluding LIoR distance)
        
        Note: LIoR distance (index 8) is expensive and computed last
        """
        N = len(candidate_coords)
        d_coord = query_coord.shape[0]
        device = query_coord.device
        
        # Initialize output
        core = torch.zeros(N, 9, device=device, dtype=query_coord.dtype)
        
        # Expand query for broadcasting
        query_exp = query_coord.unsqueeze(0).expand(N, -1)  # [N, d_coord]
        
        # (0) Cosine similarity: O(d_coord) FLOPs - on coordinates
        core[:, 0] = F.cosine_similarity(query_exp, candidate_coords, dim=1)
        
        # (1) Wedge magnitude: O(d_coord²) FLOPs (antisymmetric outer product)
        # |q ∧ c| measures rotational structure
        for i in range(N):
            outer = torch.outer(query_coord, candidate_coords[i])
            wedge = outer - outer.T  # Antisymmetric part
            core[i, 1] = torch.norm(wedge, p='fro')
        
        # (2) Tensor trace: O(d_coord) FLOPs (inner product)
        # Tr(q ⊗ c) = q · c
        core[:, 2] = torch.einsum('d,nd->n', query_coord, candidate_coords)
        
        # (3-4) Spinor magnitude and phase: O(d_spinor) FLOPs
        # Complex overlap via spinor projection - on embeddings
        psi_q = self.manifold.to_spinor(query_embedding.unsqueeze(0)).squeeze()  # [d_spinor]
        
        for i in range(N):
            psi_c = self.manifold.to_spinor(candidate_embeddings[i].unsqueeze(0)).squeeze()
            
            # Complex overlap (treat as complex numbers)
            # Split into real/imag if d_spinor is even
            if len(psi_q) % 2 == 0:
                mid = len(psi_q) // 2
                psi_q_complex = psi_q[:mid] + 1j * psi_q[mid:]
                psi_c_complex = psi_c[:mid] + 1j * psi_c[mid:]
                overlap = torch.dot(torch.conj(psi_q_complex), psi_c_complex)
            else:
                # Fallback: treat as real
                overlap = torch.dot(psi_q, psi_c) + 0j
            
            core[i, 3] = torch.abs(overlap)      # Magnitude
            core[i, 4] = torch.angle(overlap)    # Phase
        
        # (5) Energy: O(d_coord²) FLOPs (quadratic form)
        # E = q^T H c where H is the metric (acting as Hamiltonian)
        # Use base metric as the Hamiltonian
        H = self.manifold.base_metric()  # [d_coord, d_coord]
        core[:, 5] = torch.real(query_coord @ H @ candidate_coords.T)
        
        # (6-7) Tangent space distances: O(d_coord) FLOPs each
        # Compute approximate tangent vectors
        midpoints = (query_exp + candidate_coords) / 2  # [N, d_coord]
        
        for i in range(N):
            # Approximate tangent vector (simple difference from midpoint)
            v = candidate_coords[i] - midpoints[i]
            core[i, 6] = torch.norm(v)       # L2 distance
            core[i, 7] = torch.abs(v).sum()  # L1 distance
        
        # (8) LIoR distance: O(d_coord × num_samples) FLOPs (EXPENSIVE)
        # Geodesic distance (PRIMARY similarity measure)
        # Computed last as it's the most expensive
        for i in range(N):
            core[i, 8] = self.manifold.lior_distance(
                query_coord.unsqueeze(0),
                candidate_coords[i].unsqueeze(0),
                num_samples=10  # Reduced for speed (vs default 20)
            ).squeeze()
        
        return core
    
    def aggregate_to_scalar(
        self,
        similarity_vectors: torch.Tensor,  # [N, D]
        strategy: str = 'lior_primary'
    ) -> torch.Tensor:
        """
        Aggregate D-dimensional vectors to scalar scores for ranking.
        
        Args:
            similarity_vectors: Similarity vectors [N, D]
            strategy: Aggregation strategy
                - 'mean': Simple average across dimensions
                - 'lior_primary': Use LIoR distance (dim 8) as primary
                - 'weighted': Heuristic weighted combination
        
        Returns:
            Scalar scores [N]
            
        Example:
            >>> vectors = torch.randn(100, 9)
            >>> sim = ComprehensiveSimilarity(manifold, mode='core')
            >>> scores = sim.aggregate_to_scalar(vectors, strategy='lior_primary')
        """
        if strategy == 'mean':
            # Simple average (all dimensions equal weight)
            return similarity_vectors.mean(dim=1)
        
        elif strategy == 'lior_primary':
            # LIoR distance is dim 8 (last index)
            # Negative because smaller distance = higher similarity
            return -similarity_vectors[:, 8]
        
        elif strategy == 'weighted':
            # Heuristic weights based on importance
            # Normalized to sum to 1.0
            D = similarity_vectors.shape[1]
            
            if D >= 9:
                # 9D core weights
                # Indices: 0=cosine, 1=wedge, 2=tensor, 3=spinor_mag, 4=spinor_phase,
                #          5=energy, 6=l2_tangent, 7=l1_tangent, 8=lior_distance
                weights = torch.tensor([
                    0.10,  # 0: cosine (good baseline)
                    0.05,  # 1: wedge (rotation)
                    0.10,  # 2: tensor (correlation)
                    0.05,  # 3: spinor_mag (phase overlap)
                    0.05,  # 4: spinor_phase
                    0.15,  # 5: energy (important for field coupling)
                    0.05,  # 6: l2_tangent
                    0.05,  # 7: l1_tangent
                    0.40,  # 8: lior_distance (PRIMARY - geodesic)
                ], device=similarity_vectors.device, dtype=similarity_vectors.dtype)
                
                # Pad or trim weights if needed
                if D < 9:
                    weights = weights[:D]
                    weights = weights / weights.sum()  # Renormalize
                elif D > 9:
                    # Pad with zeros for extended dimensions
                    padding = torch.zeros(D - 9, device=weights.device, dtype=weights.dtype)
                    weights = torch.cat([weights, padding])
                
                return similarity_vectors @ weights
            else:
                # Fallback to mean if dimensions don't match
                return similarity_vectors.mean(dim=1)
        
        else:
            raise ValueError(
                f"Unknown aggregation strategy: {strategy}. "
                f"Must be 'mean', 'lior_primary', or 'weighted'."
            )
    
    def clear_cache(self):
        """Clear precomputed context cache."""
        self.context_cache.clear()


# Helper functions for standalone use

def compute_cosine_similarity(
    query: torch.Tensor,
    candidates: torch.Tensor
) -> torch.Tensor:
    """
    Compute cosine similarity (dimension 0 of 9D vector).
    
    Args:
        query: Query vector [d]
        candidates: Candidate vectors [N, d]
        
    Returns:
        Cosine similarities [N]
    """
    query_exp = query.unsqueeze(0).expand_as(candidates)
    return F.cosine_similarity(query_exp, candidates, dim=1)


def compute_wedge_magnitude(
    query: torch.Tensor,
    candidate: torch.Tensor
) -> torch.Tensor:
    """
    Compute wedge product magnitude (dimension 1 of 9D vector).
    
    The wedge product q ∧ c measures rotational/orthogonal structure.
    
    Args:
        query: Query vector [d]
        candidate: Candidate vector [d]
        
    Returns:
        Wedge magnitude (scalar)
    """
    outer = torch.outer(query, candidate)
    wedge = outer - outer.T  # Antisymmetric part
    return torch.norm(wedge, p='fro')


def compute_tensor_trace(
    query: torch.Tensor,
    candidates: torch.Tensor
) -> torch.Tensor:
    """
    Compute tensor product trace (dimension 2 of 9D vector).
    
    Tr(q ⊗ c) = q · c (inner product)
    
    Args:
        query: Query vector [d]
        candidates: Candidate vectors [N, d]
        
    Returns:
        Tensor traces [N]
    """
    return torch.einsum('d,nd->n', query, candidates)

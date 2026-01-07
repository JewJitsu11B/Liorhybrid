"""
Constitutive Cognitive State

A non-Markovian state representation where:
- Interactions produce bivectors that REPLACE parent vectors (not intermediate)
- Bivectors decompose into: elastic (decays), plastic (persists), excess (staging)
- Material properties track HOW the system has been stressed (not just current state)
- Phase transitions create new concepts when yield threshold exceeded

The couples analogy:
- Young healthy: high elasticity, quick recovery, minimal hysteresis
- Old healthy: stiffer, subtle changes, absorbs without phase change
- Old toxic: brittle, small perturbations cause cracks, easy phase transitions

Key insight: The same interaction does NOT mean the same update,
because the material remembers how it has been stressed before.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict, List, NamedTuple
from dataclasses import dataclass, field
import time


@dataclass
class MaterialProperties:
    """
    Non-Markovian attributes that determine how the system responds to stress.

    These are NOT state - they are response parameters that evolve over time.
    """
    # Elasticity: how much deformation relaxes back
    # High = young/healthy, Low = old/stiff
    elasticity: torch.Tensor = None  # (d_model,) per-dimension

    # Yield threshold: how much stress before plastic deformation
    # High = resilient, Low = brittle
    yield_threshold: torch.Tensor = None  # (d_model,)

    # Fatigue: accumulated stress history
    # Low = fresh, High = worn
    fatigue: torch.Tensor = None  # (d_model,)

    # Damping: how quickly oscillations settle
    damping: float = 0.1

    # Recovery rate: how fast elastic component decays
    recovery_rate: float = 0.9

    # Plastic accumulation rate: how much residual strain builds
    plastic_rate: float = 0.01


class BivectorDecomposition(NamedTuple):
    """Result of decomposing an interaction bivector."""
    elastic: torch.Tensor      # Reversible, decays toward prior
    plastic: torch.Tensor      # Small but persistent, modifies future response
    excess: torch.Tensor       # Unstable, goes to staging area
    angle: torch.Tensor        # Interaction angle (measure of novelty)
    energy: torch.Tensor       # Total interaction energy


class StagedConcept(NamedTuple):
    """A metastable concept in the staging area."""
    vector: torch.Tensor       # The excess energy vector
    parent_ids: Tuple[int, int]  # Provenance (lightweight)
    birth_time: float          # When it was created
    activation_count: int      # How many times it's been reinforced
    energy: float              # Current energy level


@dataclass
class ConstitutiveState:
    """
    Full constitutive state of the cognitive material.

    This is NOT just "what the system knows" -
    it's "how the system responds to stress given its history".
    """
    # Current state (bivector form, not parent vectors)
    state: torch.Tensor = None  # (batch, seq, d_model)

    # Material properties (non-Markovian)
    material: MaterialProperties = field(default_factory=MaterialProperties)

    # Residual strain (accumulated plastic deformation)
    residual_strain: torch.Tensor = None  # (batch, seq, d_model)

    # Staging area for candidate concepts
    staging: List[StagedConcept] = field(default_factory=list)

    # Phase history (which phase transitions have occurred)
    phase_history: List[Tuple[float, int]] = field(default_factory=list)


class ConstitutiveLayer(nn.Module):
    """
    A layer that implements constitutive cognitive mechanics.

    Instead of attention:
    - Interactions produce bivectors
    - Bivectors decompose into elastic/plastic/excess
    - Material properties determine response
    - Phase transitions create new concepts
    """

    def __init__(
        self,
        d_model: int,
        initial_elasticity: float = 0.9,
        initial_yield: float = 0.5,
        staging_capacity: int = 64,
        phase_threshold: float = 0.8,
        decay_rate: float = 0.95
    ):
        super().__init__()

        self.d_model = d_model
        self.staging_capacity = staging_capacity
        self.phase_threshold = phase_threshold
        self.decay_rate = decay_rate

        # Learnable interaction operator (produces bivector from two vectors)
        self.interact = nn.Bilinear(d_model, d_model, d_model, bias=False)

        # Decomposition projections
        self.proj_elastic = nn.Linear(d_model, d_model)
        self.proj_plastic = nn.Linear(d_model, d_model)
        self.proj_excess = nn.Linear(d_model, d_model)

        # Material property evolution
        self.elasticity_update = nn.Linear(d_model, d_model)
        self.yield_update = nn.Linear(d_model, d_model)

        # Phase transition detector
        self.phase_detector = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
            nn.Sigmoid()
        )

        # Initialize material properties (will be set on first forward)
        self.register_buffer('default_elasticity',
                           torch.full((d_model,), initial_elasticity))
        self.register_buffer('default_yield',
                           torch.full((d_model,), initial_yield))

    def compute_bivector(
        self,
        v1: torch.Tensor,
        v2: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute interaction bivector from two vectors.

        Returns:
            bivector: The interaction result
            angle: The interaction angle (cosine similarity)
        """
        # Bilinear interaction (approximates wedge product structure)
        bivector = self.interact(v1, v2)

        # Compute interaction angle
        v1_norm = v1 / (v1.norm(dim=-1, keepdim=True) + 1e-8)
        v2_norm = v2 / (v2.norm(dim=-1, keepdim=True) + 1e-8)
        angle = (v1_norm * v2_norm).sum(dim=-1, keepdim=True)

        return bivector, angle

    def decompose_bivector(
        self,
        bivector: torch.Tensor,
        angle: torch.Tensor,
        material: MaterialProperties
    ) -> BivectorDecomposition:
        """
        Decompose bivector into elastic, plastic, and excess components.

        The decomposition depends on material properties (non-Markovian).
        """
        # Get material properties
        elasticity = material.elasticity
        yield_thresh = material.yield_threshold
        fatigue = material.fatigue

        # Compute stress magnitude
        stress = bivector.norm(dim=-1, keepdim=True)

        # Elastic component: proportional to elasticity, decays
        # High elasticity = more goes to elastic (recoverable)
        elastic_frac = elasticity.unsqueeze(0).unsqueeze(0)  # broadcast
        elastic = self.proj_elastic(bivector) * elastic_frac

        # Plastic component: what exceeds yield threshold stays
        # Modified by fatigue (high fatigue = more plastic)
        fatigue_factor = 1 + fatigue.unsqueeze(0).unsqueeze(0) * 0.5
        yield_scaled = yield_thresh.unsqueeze(0).unsqueeze(0) / fatigue_factor

        plastic_mask = (stress > yield_scaled).float()
        plastic = self.proj_plastic(bivector) * plastic_mask * material.plastic_rate

        # Excess component: very high stress that can't be absorbed
        # Goes to staging area as candidate concept
        excess_thresh = yield_scaled * 2  # Double yield = excess
        excess_mask = (stress > excess_thresh).float()
        excess = self.proj_excess(bivector) * excess_mask

        # Compute total energy
        energy = stress.squeeze(-1)

        return BivectorDecomposition(
            elastic=elastic,
            plastic=plastic,
            excess=excess,
            angle=angle,
            energy=energy
        )

    def apply_elastic_decay(
        self,
        state: torch.Tensor,
        elastic: torch.Tensor,
        material: MaterialProperties
    ) -> torch.Tensor:
        """
        Apply elastic decay with hysteresis.

        Key insight: You retracted from deformation, but the elastic
        isn't quite as strong this time (hysteresis).
        """
        # Decay elastic component
        decayed = elastic * material.recovery_rate

        # Update state (elastic part relaxes back, but not fully)
        new_state = state + decayed

        return new_state

    def accumulate_plastic(
        self,
        residual: torch.Tensor,
        plastic: torch.Tensor,
        material: MaterialProperties
    ) -> Tuple[torch.Tensor, MaterialProperties]:
        """
        Accumulate plastic deformation and update material properties.

        This is where hysteresis lives - future responses change
        based on past stress.
        """
        # Accumulate residual strain
        new_residual = residual + plastic

        # Update fatigue (stress history accumulates)
        plastic_norm = plastic.norm(dim=-1).mean(dim=(0, 1))  # average over batch/seq
        material.fatigue = material.fatigue + plastic_norm * 0.01

        # Update elasticity (gets slightly lower with accumulated strain)
        # This is the "not quite as strong" effect
        strain_effect = new_residual.abs().mean(dim=(0, 1))
        material.elasticity = material.elasticity * (1 - strain_effect * 0.001)
        material.elasticity = material.elasticity.clamp(min=0.1)  # Don't go negative

        return new_residual, material

    def process_excess(
        self,
        excess: torch.Tensor,
        staging: List[StagedConcept],
        current_time: float
    ) -> List[StagedConcept]:
        """
        Process excess energy into staging area.

        Excess vectors are candidates for new concepts.
        They must stabilize (recur, attract) to become real concepts.
        """
        # Find significant excess vectors
        excess_norm = excess.norm(dim=-1)  # (batch, seq)

        # For each significant excess, create staged concept
        # (simplified: just take top-k per batch)
        batch_size = excess.shape[0]

        for b in range(batch_size):
            norms = excess_norm[b]
            top_k = min(4, norms.numel())
            top_indices = norms.topk(top_k).indices

            for idx in top_indices:
                if norms[idx] > 0.1:  # threshold
                    # Check if similar concept exists
                    vec = excess[b, idx]
                    found_match = False

                    for i, staged in enumerate(staging):
                        sim = torch.cosine_similarity(
                            vec.unsqueeze(0),
                            staged.vector.unsqueeze(0)
                        )
                        if sim > 0.8:
                            # Reinforce existing staged concept
                            staging[i] = StagedConcept(
                                vector=staged.vector * 0.9 + vec * 0.1,
                                parent_ids=staged.parent_ids,
                                birth_time=staged.birth_time,
                                activation_count=staged.activation_count + 1,
                                energy=staged.energy + norms[idx].item()
                            )
                            found_match = True
                            break

                    if not found_match and len(staging) < self.staging_capacity:
                        staging.append(StagedConcept(
                            vector=vec.detach(),
                            parent_ids=(b, idx.item()),
                            birth_time=current_time,
                            activation_count=1,
                            energy=norms[idx].item()
                        ))

        # Decay old staged concepts
        staging = [
            StagedConcept(
                vector=s.vector,
                parent_ids=s.parent_ids,
                birth_time=s.birth_time,
                activation_count=s.activation_count,
                energy=s.energy * self.decay_rate
            )
            for s in staging
            if s.energy * self.decay_rate > 0.01  # Prune weak concepts
        ]

        return staging

    def check_phase_transition(
        self,
        state: torch.Tensor,
        residual: torch.Tensor,
        staging: List[StagedConcept]
    ) -> Tuple[bool, Optional[torch.Tensor]]:
        """
        Check if a phase transition should occur.

        Phase transition = new concept becomes real (gains inertia).
        """
        # Check staging area for concepts ready to graduate
        for staged in staging:
            if staged.activation_count > 5 and staged.energy > self.phase_threshold:
                # This concept has stabilized
                # It should be incorporated into the main state
                return True, staged.vector

        return False, None

    def forward(
        self,
        x: torch.Tensor,
        constitutive_state: Optional[ConstitutiveState] = None,
        neighbors: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, ConstitutiveState]:
        """
        Forward pass through constitutive layer.

        Args:
            x: Current state vectors (batch, seq, d_model)
            constitutive_state: Previous constitutive state (or None to init)
            neighbors: Neighbor vectors to interact with (batch, seq, k, d_model)

        Returns:
            new_x: Updated state (bivector form)
            new_state: Updated constitutive state
        """
        batch, seq, d = x.shape
        device = x.device
        current_time = time.time()

        # Initialize state if needed
        if constitutive_state is None:
            constitutive_state = ConstitutiveState(
                state=x.clone(),
                material=MaterialProperties(
                    elasticity=self.default_elasticity.clone(),
                    yield_threshold=self.default_yield.clone(),
                    fatigue=torch.zeros(d, device=device)
                ),
                residual_strain=torch.zeros_like(x),
                staging=[],
                phase_history=[]
            )

        material = constitutive_state.material
        residual = constitutive_state.residual_strain
        staging = constitutive_state.staging

        # If no neighbors provided, use self-interaction (shifted)
        if neighbors is None:
            # Interact with neighbors (shifted version of self)
            x_shifted = torch.roll(x, shifts=1, dims=1)
            neighbors = x_shifted.unsqueeze(2)  # (batch, seq, 1, d)

        # Compute interactions with all neighbors
        total_elastic = torch.zeros_like(x)
        total_plastic = torch.zeros_like(x)
        total_excess = torch.zeros_like(x)

        n_neighbors = neighbors.shape[2]

        for k in range(n_neighbors):
            neighbor = neighbors[:, :, k, :]

            # Compute bivector from interaction
            bivector, angle = self.compute_bivector(x, neighbor)

            # Decompose based on material properties
            decomp = self.decompose_bivector(bivector, angle, material)

            total_elastic = total_elastic + decomp.elastic
            total_plastic = total_plastic + decomp.plastic
            total_excess = total_excess + decomp.excess

        # Average over neighbors
        total_elastic = total_elastic / n_neighbors
        total_plastic = total_plastic / n_neighbors
        total_excess = total_excess / n_neighbors

        # Apply elastic decay (with hysteresis)
        new_x = self.apply_elastic_decay(x, total_elastic, material)

        # Accumulate plastic deformation
        new_residual, new_material = self.accumulate_plastic(
            residual, total_plastic, material
        )

        # Process excess into staging area
        new_staging = self.process_excess(total_excess, staging, current_time)

        # Check for phase transition
        transition, new_concept = self.check_phase_transition(
            new_x, new_residual, new_staging
        )

        if transition and new_concept is not None:
            # Phase transition: incorporate new concept
            # This could expand the state space or modify existing structure
            constitutive_state.phase_history.append((current_time, len(new_staging)))
            # For now, just add to residual (could be more sophisticated)
            new_residual = new_residual + new_concept.unsqueeze(0).unsqueeze(0) * 0.1

        # Update constitutive state
        new_state = ConstitutiveState(
            state=new_x,
            material=new_material,
            residual_strain=new_residual,
            staging=new_staging,
            phase_history=constitutive_state.phase_history
        )

        # Final output includes residual strain (non-Markovian effect)
        output = new_x + new_residual * 0.1  # Residual influences but doesn't dominate

        return output, new_state


class ConstitutiveStack(nn.Module):
    """
    Stack of constitutive layers with shared material state.

    This replaces the attention-based stack with a constitutive
    mechanics-based approach.
    """

    def __init__(
        self,
        d_model: int,
        n_layers: int = 4,
        initial_elasticity: float = 0.9,
        initial_yield: float = 0.5
    ):
        super().__init__()

        self.layers = nn.ModuleList([
            ConstitutiveLayer(
                d_model=d_model,
                initial_elasticity=initial_elasticity,
                initial_yield=initial_yield
            )
            for _ in range(n_layers)
        ])

        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        constitutive_state: Optional[ConstitutiveState] = None
    ) -> Tuple[torch.Tensor, ConstitutiveState]:
        """Forward through all layers."""

        state = constitutive_state

        for layer in self.layers:
            x, state = layer(x, state)
            x = self.norm(x)

        return x, state

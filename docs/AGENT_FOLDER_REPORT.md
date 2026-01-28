# Folder Agent Dispatch

Three “agents” per main folder:
- **SOTA Attention Agent**: compares against standard attention/transformer baselines.
- **JAMBA Agent**: contrasts with AI21’s JAMBA hybrid (MoE + Mamba SSM).
- **Real-World Value Agent**: summarizes practical impact and deployment value.

## core/
- SOTA Attention: Lacks field-theoretic state; attention O(N²) token similarity vs core’s cognitive tensor field with geometric evolution and fractional memory.
- JAMBA: JAMBA SSM is empirical O(N) state expansion; core uses physics-grounded field dynamics with constant-memory recurrence.
- Real-World Value: Provides physically consistent state evolution for reasoning/simulation domains (science, robotics), not just text.

## kernels/
- SOTA Attention: No analogs to fractional or Hamiltonian kernels; attention weights are heuristic softmax.
- JAMBA: SSM kernels are learned filters; kernels/ offer Hamiltonian, Bayesian, fractional memory operators with stability controls.
- Real-World Value: Long-range memory and energy-aware dynamics enable faithful temporal reasoning with bounded compute.

## models/
- SOTA Attention: Standard transformers rely on positional encodings; models/ integrates geometric stacks and LIoR (Learned Inference over Riemannian) kernels with manifold-aware metrics.
- JAMBA: Hybrid MoE + Mamba mixes attention and SSM but lacks shared physical metric; models/ preserves a unified geometry.
- Real-World Value: Supports structured reasoning and multimodal alignment under one metric, aiding scientific and spatial tasks.

## training/
- SOTA Attention: Typical trainers focus on loss/optimizer; this folder enforces device invariants, geometric costs, and free-to-nudged-phase contrastive physics loops.
- JAMBA: Training is tuned for hybrid blocks; here training keeps geometry intact (frames, curvature, rotors) rather than attention/SSM heuristics.
- Real-World Value: Stable physics-aware training reduces drift and improves interpretability for regulated or safety-critical workflows.

## inference/
- SOTA Attention: Inference mirrors training but often diverges for efficiency; inference/ keeps O(N) geometric stack consistent with training kernels.
- JAMBA: JAMBA compresses into SSM for inference; here the same fractional/geometry stack is reused to avoid train/infer mismatch.
- Real-World Value: Predictable deployment behavior with the same manifold and memory properties as training, simplifying validation.

## multimodal_heads/
- SOTA Attention: Separate encoders per modality with attention fusion; these heads share the physics framework for coherent cross-modal geometry.
- JAMBA: Hybrid blocks can host multimodal inputs but lack a unified metric; here fusion respects manifold structure across modalities.
- Real-World Value: Better alignment of vision/audio/text through consistent geometry, reducing brittle modality-specific hacks.

## utils/
- SOTA Attention: Minimal provenance/auditing; utils add audit hooks and cost estimators tailored to physics stack.
- JAMBA: No per-file pipeline audit; audit records first-touch participation for integrity.
- Real-World Value: Traceability and sizing help operators validate runs and resource plans.

## configs/
- SOTA Attention: Configs for scaling attention parameters; these configs encode geometry/memory toggles and device safety.
- JAMBA: Hybrid configs center on MoE/SSM routing; here configs carry physics modes and audits.
- Real-World Value: Faster, safer bring-up with predefined physics-aware switches for experiments and ops.

## scripts/
- SOTA Attention: Scripts typically launch attention-centric training; these scripts wire physics modes and audits.
- JAMBA: Hybrid scripts focus on MoE/SSM orchestration; here scripts preserve geometry/memory settings end-to-end.
- Real-World Value: Operational recipes to reproduce physics-aware runs with minimal manual wiring.

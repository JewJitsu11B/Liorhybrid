"""
Geometric Inference Example

Demonstrates the full cognitive architecture:
1. Field evolution (unconscious processing)
2. Geometric transformer query (conscious attention)
3. Output generation (response)

This shows how the physics-based field simulation connects
to modern transformer inference via geometric attention.

Architecture Flow:
    User Prompt → Q (query embedding)
    Field T_ij(x,t) → K, V (memory extraction)
    Geometric Attention(Q, K, V, T_field) → Output
    Output → Response Tokens

The field acts as a "subconscious mind" that continuously evolves,
and the transformer acts as "conscious attention" that queries it.
"""
try: import usage_tracker; usage_tracker.track(__file__)
except: pass

import torch
import argparse
from Liorhybrid.core import CognitiveTensorField, FAST_TEST_CONFIG
from Liorhybrid.inference import GeometricTransformer


def simulate_prompt_embedding(prompt: str, d_model: int, device: torch.device) -> torch.Tensor:
    """
    Simulate prompt embedding (in real system, would use tokenizer + embedding layer).

    For demonstration, creates random embeddings with length based on prompt.

    Args:
        prompt: Input text prompt
        d_model: Embedding dimension
        device: Torch device

    Returns:
        Embedded prompt (1, seq_len, d_model)
    """
    # Simple tokenization: split by spaces
    tokens = prompt.split()
    seq_len = len(tokens)

    # Random embeddings (in real system: learned token embeddings)
    # Use hash of prompt for reproducibility
    seed = hash(prompt) % (2**32)
    torch.manual_seed(seed)

    embeddings = torch.randn(1, seq_len, d_model, device=device)

    return embeddings


def decode_output(output: torch.Tensor) -> str:
    """
    Decode transformer output to text (simulated).

    In real system, would use language model head + detokenizer.

    Args:
        output: Transformer output (1, seq_len, d_model)

    Returns:
        Decoded text string
    """
    # For demonstration, compute simple statistics of output
    seq_len = output.shape[1]
    output_norm = torch.norm(output, dim=-1).squeeze().tolist()

    # Create pseudo-response based on output magnitudes
    response_parts = []
    for i, norm in enumerate(output_norm):
        if isinstance(norm, float):
            if norm > 2.0:
                response_parts.append("[HIGH_ACTIVATION]")
            elif norm > 1.0:
                response_parts.append("[MEDIUM_ACTIVATION]")
            else:
                response_parts.append("[LOW_ACTIVATION]")

    return " ".join(response_parts)


def main():
    parser = argparse.ArgumentParser(
        description='Geometric inference with cognitive field'
    )
    parser.add_argument(
        '--prompt',
        type=str,
        default='Why is decision-making difficult under uncertainty?',
        help='Input prompt to query the field'
    )
    parser.add_argument(
        '--evolution-steps',
        type=int,
        default=100,
        help='Number of field evolution steps before querying'
    )
    parser.add_argument(
        '--adaptive',
        action='store_true',
        help='Enable adaptive parameter learning in field'
    )
    parser.add_argument(
        '--d-model',
        type=int,
        default=64,
        help='Transformer hidden dimension'
    )
    parser.add_argument(
        '--n-heads',
        type=int,
        default=4,
        help='Number of attention heads'
    )
    parser.add_argument(
        '--n-layers',
        type=int,
        default=2,
        help='Number of transformer layers'
    )
    args = parser.parse_args()

    print("=" * 70)
    print("Geometric Inference with Bayesian Cognitive Field")
    print("=" * 70)

    # 1. Initialize cognitive field
    print("\n1. Initializing cognitive field...")
    config = FAST_TEST_CONFIG
    if args.adaptive:
        config.adaptive_learning = True
        print("   Mode: Adaptive parameter learning")
    else:
        print("   Mode: Fixed parameters")

    field = CognitiveTensorField(config)
    print(f"   Field shape: {field.T.shape}")
    print(f"   Tensor dimension: {config.tensor_dim}")
    print(f"   Spatial grid: {config.spatial_size}")

    # 2. Evolve field (unconscious processing)
    print(f"\n2. Evolving field for {args.evolution_steps} steps...")
    print("   (This simulates background 'unconscious' processing)")

    initial_entropy = field.compute_entropy().item() if args.adaptive else None

    for step in range(args.evolution_steps):
        field.evolve_step()

        if (step + 1) % 25 == 0:
            norm = field.get_norm_squared()
            if args.adaptive:
                entropy = field.compute_entropy().item()
                print(f"   Step {step+1:3d}: ||T||² = {norm:.2f}, H = {entropy:.2e}")
            else:
                print(f"   Step {step+1:3d}: ||T||² = {norm:.2f}")

    if args.adaptive:
        final_entropy = field.compute_entropy().item()
        entropy_reduction = (initial_entropy - final_entropy) / initial_entropy * 100
        print(f"\n   Entropy reduction: {entropy_reduction:.1f}%")

    # 3. Initialize geometric transformer
    print("\n3. Initializing geometric transformer...")
    transformer = GeometricTransformer(
        field_dim=config.tensor_dim,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        use_positional_encoding=True,
        use_temporal_encoding=True
    )

    print(f"   Model dimension: {args.d_model}")
    print(f"   Attention heads: {args.n_heads}")
    print(f"   Transformer layers: {args.n_layers}")

    # Count parameters
    n_params = sum(p.numel() for p in transformer.parameters())
    print(f"   Total parameters: {n_params:,}")

    # 4. Embed prompt (conscious query)
    print(f"\n4. Processing prompt: \"{args.prompt}\"")
    Q_input = simulate_prompt_embedding(args.prompt, args.d_model, field.device)
    print(f"   Prompt tokens: {Q_input.shape[1]}")
    print(f"   Embedding dimension: {Q_input.shape[2]}")

    # 5. Query field via geometric attention
    print("\n5. Querying field with geometric attention...")
    print("   Computing K, V from field state...")
    print("   Applying wedge + tensor + spinor products...")
    print("   Softmax normalization...")

    with torch.no_grad():
        output, attn_weights_list = transformer(
            Q_input,
            field.T,
            time=field.t
        )

    print(f"   Output shape: {output.shape}")
    print(f"   Attention layers: {len(attn_weights_list)}")

    # Analyze attention weights
    print("\n6. Analyzing attention patterns...")
    for layer_idx, attn_weights in enumerate(attn_weights_list):
        # attn_weights: (batch, n_heads, seq_len_q, seq_len_k)
        max_weight = torch.max(attn_weights).item()
        min_weight = torch.min(attn_weights).item()
        mean_weight = torch.mean(attn_weights).item()
        entropy_attn = -torch.sum(attn_weights * torch.log(attn_weights + 1e-9), dim=-1).mean().item()

        print(f"   Layer {layer_idx + 1}:")
        print(f"      Attention range: [{min_weight:.4f}, {max_weight:.4f}]")
        print(f"      Mean attention: {mean_weight:.4f}")
        print(f"      Attention entropy: {entropy_attn:.4f}")

    # 7. Decode output (simulated)
    print("\n7. Generating response...")
    response = decode_output(output)
    print(f"   Response: {response}")

    # 8. Show geometric product contributions
    print("\n8. Geometric product analysis...")
    print("   (Showing learned weights for wedge, tensor, spinor)")

    for name, param in transformer.named_parameters():
        if 'geometric_weights' in name:
            weights = param.data
            print(f"   {name}:")
            print(f"      Wedge weight:  {weights[0].item():.4f}")
            print(f"      Tensor weight: {weights[1].item():.4f}")
            print(f"      Spinor weight: {weights[2].item():.4f}")

    # 9. Summary
    print("\n" + "=" * 70)
    print("Summary:")
    print("=" * 70)
    print(f"Field evolution: {args.evolution_steps} steps at t={field.t:.4f}")
    print(f"Final field norm: {field.get_norm_squared():.2f}")
    if args.adaptive:
        print(f"Final entropy: {field.compute_entropy().item():.2e}")
        print(f"Final α: {field.alpha.item():.4f}")
        print(f"Final ν (mean): {field.nu.mean().item():.4f}")
    print(f"Prompt length: {Q_input.shape[1]} tokens")
    print(f"Output dimension: {output.shape}")
    print(f"Memory tokens: {64} (8×8 spatial grid)")
    print("\nThe cognitive field acts as a dynamic memory that is queried")
    print("by geometric attention to produce context-aware responses.")
    print("=" * 70)


if __name__ == "__main__":
    main()

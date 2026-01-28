"""
Standalone torch.compile test for GeometricTransformerWithMamba.

Tests:
1. Compatibility - does it compile without errors/graph breaks?
2. Memory profiling - VRAM comparison eager vs compiled
"""
try: import usage_tracker; usage_tracker.track(__file__)
except: pass

import torch
import gc
from Liorhybrid.inference import GeometricTransformerWithMamba
from Liorhybrid.core import CognitiveTensorField, FieldConfig
from Liorhybrid.training.embeddings import MultimodalEmbedding


def get_memory_mb():
    """Get current GPU memory usage in MB."""
    torch.cuda.synchronize()
    return torch.cuda.memory_allocated() / 1024 / 1024


def get_peak_memory_mb():
    """Get peak GPU memory usage in MB."""
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() / 1024 / 1024


def reset_memory_stats():
    """Reset peak memory tracking."""
    torch.cuda.reset_peak_memory_stats()
    gc.collect()
    torch.cuda.empty_cache()


def test_compatibility(model, x, field):
    """Check for graph breaks and compilation errors."""
    print("\n[1] Compatibility Check")
    print("-" * 40)

    # Enable graph break logging
    torch._dynamo.config.verbose = True
    torch._dynamo.config.log_level = 20  # INFO

    try:
        compiled = torch.compile(model, mode="reduce-overhead", fullgraph=False)

        # Forward pass
        with torch.no_grad():
            output, _ = compiled(x, field.T, time=field.t)

        print(f"  ✓ Compilation succeeded")
        print(f"  Output shape: {output.shape}")

        # Check for graph breaks
        explain = torch._dynamo.explain(model)(x, field.T, time=field.t)
        print(f"  Graph breaks: {explain.graph_break_count}")
        if explain.graph_break_count > 0:
            print(f"  Break reasons:")
            for reason in explain.break_reasons[:5]:  # First 5
                print(f"    - {reason}")

        return True

    except Exception as e:
        print(f"  ✗ Compilation FAILED: {e}")
        return False

    finally:
        torch._dynamo.config.verbose = False


def test_memory(model, x, field):
    """Compare memory usage: eager vs compiled."""
    print("\n[2] Memory Profiling")
    print("-" * 40)

    results = {}

    # Eager mode
    reset_memory_stats()
    baseline_mem = get_memory_mb()

    with torch.no_grad():
        for _ in range(5):
            _ = model(x, field.T, time=field.t)

    eager_peak = get_peak_memory_mb()
    eager_current = get_memory_mb()
    results['eager'] = {
        'baseline': baseline_mem,
        'peak': eager_peak,
        'current': eager_current
    }

    print(f"  Eager mode:")
    print(f"    Baseline:  {baseline_mem:.1f} MB")
    print(f"    Peak:      {eager_peak:.1f} MB")
    print(f"    Current:   {eager_current:.1f} MB")

    # Compiled mode
    torch.compiler.reset()
    reset_memory_stats()
    baseline_mem = get_memory_mb()

    compiled = torch.compile(model, mode="reduce-overhead")

    # Compilation pass
    with torch.no_grad():
        _ = compiled(x, field.T, time=field.t)
    compile_peak = get_peak_memory_mb()

    # Reset and measure steady state
    reset_memory_stats()
    baseline_mem = get_memory_mb()

    with torch.no_grad():
        for _ in range(5):
            _ = compiled(x, field.T, time=field.t)

    compiled_peak = get_peak_memory_mb()
    compiled_current = get_memory_mb()
    results['compiled'] = {
        'compile_peak': compile_peak,
        'baseline': baseline_mem,
        'peak': compiled_peak,
        'current': compiled_current
    }

    print(f"  Compiled mode (reduce-overhead):")
    print(f"    Compile peak: {compile_peak:.1f} MB")
    print(f"    Baseline:     {baseline_mem:.1f} MB")
    print(f"    Peak:         {compiled_peak:.1f} MB")
    print(f"    Current:      {compiled_current:.1f} MB")

    # Summary
    print(f"\n  Memory delta (compiled vs eager):")
    peak_delta = results['compiled']['peak'] - results['eager']['peak']
    print(f"    Peak: {peak_delta:+.1f} MB ({'+' if peak_delta > 0 else ''}{100*peak_delta/results['eager']['peak']:.1f}%)")

    return results


def main():
    device = 'cuda'

    print("=" * 50)
    print("torch.compile Test Suite")
    print("=" * 50)

    # Small config for testing
    d_model = 256
    seq_len = 128
    batch_size = 4
    vocab_size = 1024

    # Create field
    field_config = FieldConfig(
        spatial_size=(8, 8),
        tensor_dim=16,
        dt=0.005,
        device=device
    )
    field = CognitiveTensorField(field_config)

    # Create model
    model = GeometricTransformerWithMamba(
        d_model=d_model,
        n_mamba_layers=2,
        n_attention_layers=1,
        n_heads=4,
        field_dim=16,
        use_dpr=False,
        use_positional_encoding=True,
        timing_debug=False
    ).to(device).eval()

    # Create embedding
    embedding = MultimodalEmbedding(
        vocab_size=vocab_size,
        d_model=d_model,
        max_seq_len=seq_len
    ).to(device).eval()

    # Dummy input
    tokens = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    x = embedding(tokens, modality='text')

    print(f"\nConfig: d_model={d_model}, seq={seq_len}, batch={batch_size}")
    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")

    # Run tests
    compat_ok = test_compatibility(model, x, field)

    if compat_ok:
        torch.compiler.reset()
        test_memory(model, x, field)

    print("\n" + "=" * 50)
    print("Done")


if __name__ == "__main__":
    main()

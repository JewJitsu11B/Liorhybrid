"""
Example Usage of MoE Framework

Demonstrates basic usage, training, and knowledge graph querying.
"""

import torch
import torch.nn as nn
from moe_framework import MoEConfig, FullyOptimizedMoESystem


def example_basic_usage():
    """Basic MoE system usage."""
    print("=" * 60)
    print("Example 1: Basic MoE System")
    print("=" * 60)
    
    # Create configuration
    config = MoEConfig(
        input_dim=512,
        hidden_dim=2048,
        output_dim=512,
        num_experts=8,
        top_k_experts=2,
        use_amp=False,  # Disable for simplicity in example
        use_compile=False,  # Disable for simplicity
    )
    
    # Initialize system
    model = FullyOptimizedMoESystem(config)
    if torch.cuda.is_available():
        model = model.cuda()
        print("✓ Model initialized on CUDA")
    else:
        print("✓ Model initialized on CPU")
    
    # Create sample input
    batch_size, seq_len = 4, 128
    x = torch.randn(batch_size, seq_len, config.input_dim)
    if torch.cuda.is_available():
        x = x.cuda()
    
    # Forward pass
    model.eval()
    with torch.no_grad():
        output = model(x)
    
    print(f"✓ Forward pass completed")
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {output.shape}")
    print()


def example_training():
    """Example training loop."""
    print("=" * 60)
    print("Example 2: Training Loop")
    print("=" * 60)
    
    # Configuration
    config = MoEConfig(
        input_dim=256,
        hidden_dim=1024,
        output_dim=256,
        num_experts=8,
        top_k_experts=2,
        use_amp=True,
        use_gradient_checkpointing=True,
    )
    
    # Initialize
    model = FullyOptimizedMoESystem(config)
    if torch.cuda.is_available():
        model = model.cuda()
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    loss_fn = nn.MSELoss()
    
    # Training loop
    model.train()
    num_steps = 5
    
    print(f"Training for {num_steps} steps...")
    for step in range(num_steps):
        # Create dummy batch
        x = torch.randn(8, 64, config.input_dim)
        targets = torch.randn(8, 64, config.output_dim)
        
        if torch.cuda.is_available():
            x, targets = x.cuda(), targets.cuda()
        
        # Training step
        loss = model.training_step((x, targets), optimizer, loss_fn)
        
        print(f"  Step {step + 1}/{num_steps}: Loss = {loss:.4f}")
    
    print("✓ Training completed")
    print()


def example_knowledge_graph_query():
    """Example knowledge graph querying."""
    print("=" * 60)
    print("Example 3: Knowledge Graph Querying")
    print("=" * 60)
    
    # Configuration
    config = MoEConfig(
        input_dim=256,
        hidden_dim=512,
        output_dim=256,
        num_experts=4,
        top_k_experts=2,
    )
    
    # Initialize
    model = FullyOptimizedMoESystem(config)
    if torch.cuda.is_available():
        model = model.cuda()
    
    model.eval()
    
    # Process some inputs to populate knowledge graph
    print("Populating knowledge graph...")
    with torch.no_grad():
        for i in range(3):
            x = torch.randn(4, 32, config.input_dim)
            if torch.cuda.is_available():
                x = x.cuda()
            _ = model(x, use_knowledge_graph=True)
    
    num_nodes = model.moe.knowledge_graph.num_nodes
    print(f"✓ Knowledge graph populated with {num_nodes} nodes")
    
    # Query knowledge graph
    if num_nodes > 0:
        print("\nQuerying knowledge graph...")
        query = torch.randn(config.output_dim)
        if torch.cuda.is_available():
            query = query.cuda()
        
        results = model.query_knowledge_graph(query, top_k=3)
        
        print(f"✓ Found {len(results)} similar nodes:")
        for i, result in enumerate(results):
            print(f"  {i+1}. Node {result['node_id']}: "
                  f"Similarity={result['similarity']:.3f}, "
                  f"Specialization={result['metadata'].get('specialization', 'N/A')}")
    print()


def example_cuda_optimizations():
    """Example with CUDA optimizations enabled."""
    print("=" * 60)
    print("Example 4: CUDA Optimizations")
    print("=" * 60)
    
    if not torch.cuda.is_available():
        print("CUDA not available, skipping this example")
        return
    
    # Configuration with all optimizations
    config = MoEConfig(
        input_dim=512,
        hidden_dim=2048,
        output_dim=512,
        num_experts=16,
        top_k_experts=3,
        use_compile=True,
        compile_mode='reduce-overhead',  # Fast compile
        use_amp=True,
        use_gradient_checkpointing=True,
    )
    
    # Initialize
    model = FullyOptimizedMoESystem(config).cuda()
    model.eval()
    
    # Benchmark
    print("Running benchmark...")
    x = torch.randn(16, 128, config.input_dim).cuda()
    
    # Warmup
    with torch.no_grad():
        for _ in range(3):
            _ = model(x)
    
    # Measure
    import time
    num_iterations = 10
    
    torch.cuda.synchronize()
    start = time.time()
    
    with torch.no_grad():
        for _ in range(num_iterations):
            _ = model(x)
    
    torch.cuda.synchronize()
    elapsed = time.time() - start
    
    throughput = (num_iterations * x.size(0)) / elapsed
    
    print(f"✓ Benchmark completed")
    print(f"  Iterations: {num_iterations}")
    print(f"  Total time: {elapsed:.3f}s")
    print(f"  Throughput: {throughput:.1f} samples/sec")
    print()


def main():
    """Run all examples."""
    print("\n" + "=" * 60)
    print("MoE Framework Examples")
    print("=" * 60 + "\n")
    
    try:
        example_basic_usage()
        example_training()
        example_knowledge_graph_query()
        example_cuda_optimizations()
        
        print("=" * 60)
        print("All examples completed successfully!")
        print("=" * 60)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()

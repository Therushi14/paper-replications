"""
Example usage of the GPT-3 implementation.
This script demonstrates how to use the GPT-3 model for basic tasks.
"""

import torch
from gpt3_pytorch import GPT3, get_gpt3_config, count_parameters


def example_basic_usage():
    """Example 1: Basic model creation and forward pass."""
    print("Example 1: Basic Usage")
    print("=" * 70)
    
    # Create a small GPT-3 model
    config = get_gpt3_config('small')
    model = GPT3(**config)
    
    print(f"Model: GPT-3 Small")
    print(f"Parameters: {count_parameters(model):,}")
    print(f"Max sequence length: {config['max_seq_len']}")
    print(f"Vocabulary size: {config['vocab_size']}")
    print()
    
    # Create sample input
    batch_size = 2
    seq_len = 32
    input_ids = torch.randint(0, config['vocab_size'], (batch_size, seq_len))
    
    # Forward pass
    logits = model(input_ids)
    print(f"Input shape: {input_ids.shape}")
    print(f"Output shape: {logits.shape}")
    print(f"Logits range: [{logits.min():.2f}, {logits.max():.2f}]")
    print()


def example_training_step():
    """Example 2: Simulated training step."""
    print("Example 2: Training Step")
    print("=" * 70)
    
    config = get_gpt3_config('small')
    model = GPT3(**config)
    
    # Create an optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    
    # Create sample data
    batch_size = 4
    seq_len = 64
    input_ids = torch.randint(0, config['vocab_size'], (batch_size, seq_len))
    targets = torch.randint(0, config['vocab_size'], (batch_size, seq_len))
    
    # Training step
    model.train()
    optimizer.zero_grad()
    
    logits, loss = model(input_ids, targets)
    loss.backward()
    optimizer.step()
    
    print(f"Batch size: {batch_size}")
    print(f"Sequence length: {seq_len}")
    print(f"Loss: {loss.item():.4f}")
    print(f"Perplexity: {torch.exp(loss).item():.2f}")
    print()


def example_text_generation():
    """Example 3: Text generation with different sampling strategies."""
    print("Example 3: Text Generation")
    print("=" * 70)
    
    config = get_gpt3_config('small')
    model = GPT3(**config)
    model.eval()
    
    # Start with a sequence
    start_tokens = torch.randint(0, config['vocab_size'], (1, 10))
    print(f"Starting with {start_tokens.shape[1]} tokens")
    print()
    
    # Generate with different temperatures
    temperatures = [0.5, 1.0, 1.5]
    
    with torch.no_grad():
        for temp in temperatures:
            generated = model.generate(
                start_tokens.clone(),
                max_new_tokens=20,
                temperature=temp,
                top_k=40
            )
            print(f"Temperature {temp}: Generated {generated.shape[1]} tokens total")
    
    print()


def example_model_comparison():
    """Example 4: Compare different model sizes."""
    print("Example 4: Model Size Comparison")
    print("=" * 70)
    
    sizes = ['small', 'medium', 'large']
    
    print(f"{'Model':<10} {'Parameters':>15} {'Layers':>7} {'d_model':>8} {'n_heads':>8}")
    print("-" * 70)
    
    for size in sizes:
        config = get_gpt3_config(size)
        model = GPT3(**config)
        params = count_parameters(model)
        
        print(f"{size.upper():<10} {params:>15,} {config['n_layers']:>7} "
              f"{config['d_model']:>8} {config['n_heads']:>8}")
    
    print()


def example_inference_optimization():
    """Example 5: Optimized inference mode."""
    print("Example 5: Inference Optimization")
    print("=" * 70)
    
    config = get_gpt3_config('small')
    model = GPT3(**config)
    model.eval()
    
    # Put in inference mode (disable dropout, etc.)
    with torch.no_grad():
        input_ids = torch.randint(0, config['vocab_size'], (1, 50))
        
        # Generate without computing gradients
        generated = model.generate(
            input_ids,
            max_new_tokens=30,
            temperature=0.8,
            top_k=50
        )
        
        print(f"Input length: {input_ids.shape[1]}")
        print(f"Generated length: {generated.shape[1]}")
        print(f"New tokens: {generated.shape[1] - input_ids.shape[1]}")
        print("✓ Generation completed in inference mode (no gradients)")
    
    print()


def example_custom_config():
    """Example 6: Create a model with custom configuration."""
    print("Example 6: Custom Configuration")
    print("=" * 70)
    
    # Create a custom small model for experimentation
    custom_config = {
        'vocab_size': 10000,  # Smaller vocabulary
        'max_seq_len': 512,    # Shorter sequences
        'd_model': 256,        # Smaller embedding
        'n_layers': 6,         # Fewer layers
        'n_heads': 8,          # Fewer heads
        'd_ff': 1024,          # Smaller FFN
        'dropout': 0.1
    }
    
    model = GPT3(**custom_config)
    params = count_parameters(model)
    
    print(f"Custom Configuration:")
    print(f"  Vocabulary size: {custom_config['vocab_size']}")
    print(f"  Max sequence length: {custom_config['max_seq_len']}")
    print(f"  Model dimension: {custom_config['d_model']}")
    print(f"  Number of layers: {custom_config['n_layers']}")
    print(f"  Number of heads: {custom_config['n_heads']}")
    print(f"  Total parameters: {params:,}")
    
    # Test it
    input_ids = torch.randint(0, custom_config['vocab_size'], (2, 32))
    logits = model(input_ids)
    print(f"\n✓ Custom model works: {input_ids.shape} -> {logits.shape}")
    
    print()


def main():
    """Run all examples."""
    print("\n")
    print("╔" + "═" * 68 + "╗")
    print("║" + " " * 15 + "GPT-3 Implementation Examples" + " " * 23 + "║")
    print("╚" + "═" * 68 + "╝")
    print()
    
    example_basic_usage()
    example_training_step()
    example_text_generation()
    example_model_comparison()
    example_inference_optimization()
    example_custom_config()
    
    print("=" * 70)
    print("All examples completed successfully!")
    print("=" * 70)
    print()


if __name__ == "__main__":
    main()

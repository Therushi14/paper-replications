"""
Simple test script to validate the GPT-3 implementation.
This script runs basic tests to ensure all components work correctly.
"""

import torch
from gpt3_pytorch import GPT3, get_gpt3_config, count_parameters


def test_forward_pass():
    """Test basic forward pass."""
    print("Test 1: Basic Forward Pass")
    print("-" * 60)
    
    config = get_gpt3_config('small')
    model = GPT3(**config)
    
    batch_size = 4
    seq_len = 64
    input_ids = torch.randint(0, config['vocab_size'], (batch_size, seq_len))
    
    # Test without targets
    logits = model(input_ids)
    assert logits.shape == (batch_size, seq_len, config['vocab_size'])
    print(f"✓ Forward pass without targets: {input_ids.shape} -> {logits.shape}")
    
    # Test with targets (compute loss)
    targets = torch.randint(0, config['vocab_size'], (batch_size, seq_len))
    logits, loss = model(input_ids, targets)
    assert logits.shape == (batch_size, seq_len, config['vocab_size'])
    assert loss is not None
    print(f"✓ Forward pass with targets: Loss = {loss.item():.4f}")
    print()


def test_generation():
    """Test text generation."""
    print("Test 2: Text Generation")
    print("-" * 60)
    
    config = get_gpt3_config('small')
    model = GPT3(**config)
    model.eval()
    
    # Start with a short sequence
    input_ids = torch.randint(0, config['vocab_size'], (1, 5))
    print(f"Starting sequence length: {input_ids.shape[1]}")
    
    # Generate tokens
    with torch.no_grad():
        generated = model.generate(input_ids, max_new_tokens=10, temperature=1.0)
    
    assert generated.shape[1] == input_ids.shape[1] + 10
    print(f"✓ Generated sequence length: {generated.shape[1]}")
    print(f"✓ Generated {generated.shape[1] - input_ids.shape[1]} new tokens")
    print()


def test_different_sizes():
    """Test different model sizes."""
    print("Test 3: Different Model Sizes")
    print("-" * 60)
    
    sizes = ['small', 'medium']  # Testing small sizes only for speed
    
    for size in sizes:
        config = get_gpt3_config(size)
        model = GPT3(**config)
        params = count_parameters(model)
        
        # Test forward pass
        input_ids = torch.randint(0, config['vocab_size'], (1, 16))
        logits = model(input_ids)
        
        print(f"✓ GPT-3 {size.upper()}: {params:,} params - Forward pass successful")
    
    print()


def test_attention_mask():
    """Test causal attention masking."""
    print("Test 4: Causal Attention Masking")
    print("-" * 60)
    
    config = get_gpt3_config('small')
    model = GPT3(**config)
    
    seq_len = 8
    mask = model.create_causal_mask(seq_len, torch.device('cpu'))
    
    # Check mask is lower triangular
    expected_mask = torch.tril(torch.ones(seq_len, seq_len))
    assert torch.all(mask.squeeze() == expected_mask)
    print(f"✓ Causal mask is correctly lower triangular (shape: {mask.shape})")
    print("Mask visualization (first 5x5):")
    print(mask[0, 0, :5, :5].int())
    print()


def test_weight_tying():
    """Test weight tying between embeddings and output layer."""
    print("Test 5: Weight Tying")
    print("-" * 60)
    
    config = get_gpt3_config('small')
    model = GPT3(**config)
    
    # Check that weights are tied
    assert model.token_embedding.weight is model.lm_head.weight
    print("✓ Token embedding and LM head weights are tied")
    print()


def test_gradient_flow():
    """Test that gradients flow through the model."""
    print("Test 6: Gradient Flow")
    print("-" * 60)
    
    config = get_gpt3_config('small')
    model = GPT3(**config)
    
    batch_size = 2
    seq_len = 16
    input_ids = torch.randint(0, config['vocab_size'], (batch_size, seq_len))
    targets = torch.randint(0, config['vocab_size'], (batch_size, seq_len))
    
    # Forward and backward pass
    logits, loss = model(input_ids, targets)
    loss.backward()
    
    # Check that gradients exist
    has_gradients = any(p.grad is not None for p in model.parameters() if p.requires_grad)
    assert has_gradients
    print("✓ Gradients flow correctly through the model")
    print(f"✓ Loss: {loss.item():.4f}")
    print()


def main():
    print("=" * 60)
    print("GPT-3 Implementation Tests")
    print("=" * 60)
    print()
    
    # Run all tests
    test_forward_pass()
    test_generation()
    test_different_sizes()
    test_attention_mask()
    test_weight_tying()
    test_gradient_flow()
    
    print("=" * 60)
    print("All tests passed! ✓")
    print("=" * 60)


if __name__ == "__main__":
    main()

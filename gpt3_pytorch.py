"""
GPT-3: Language Models are Few-Shot Learners
Paper: https://arxiv.org/abs/2005.14165
Authors: Tom B. Brown et al. (OpenAI)
Year: 2020

This is a clean PyTorch implementation of the GPT-3 architecture from scratch.
GPT-3 is an autoregressive language model based on the Transformer decoder architecture.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Self-Attention mechanism as described in "Attention is All You Need"
    and used in GPT-3.
    
    Args:
        d_model: Dimension of the model (embedding dimension)
        n_heads: Number of attention heads
        dropout: Dropout probability
    """
    
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        # Linear projections for Q, K, V
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        
        # Output projection
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            mask: Optional mask tensor of shape (batch_size, seq_len, seq_len)
        
        Returns:
            Output tensor of shape (batch_size, seq_len, d_model)
        """
        batch_size, seq_len, d_model = x.shape
        
        # Linear projections and reshape to (batch_size, n_heads, seq_len, d_k)
        Q = self.w_q(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.w_k(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = self.w_v(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        
        # Scaled dot-product attention
        # scores shape: (batch_size, n_heads, seq_len, seq_len)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # Apply mask (for causal/autoregressive attention)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # Apply softmax and dropout
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        # output shape: (batch_size, n_heads, seq_len, d_k)
        output = torch.matmul(attention_weights, V)
        
        # Concatenate heads and apply output projection
        # Reshape to (batch_size, seq_len, d_model)
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        output = self.w_o(output)
        
        return output


class FeedForward(nn.Module):
    """
    Position-wise Feed-Forward Network used in Transformer blocks.
    
    Args:
        d_model: Dimension of the model
        d_ff: Dimension of the feed-forward network (usually 4 * d_model)
        dropout: Dropout probability
    """
    
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
        
        Returns:
            Output tensor of shape (batch_size, seq_len, d_model)
        """
        # GELU activation is used in GPT models
        x = self.linear1(x)
        x = F.gelu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x


class TransformerBlock(nn.Module):
    """
    A single Transformer decoder block as used in GPT-3.
    Consists of masked multi-head attention followed by feed-forward network,
    with layer normalization and residual connections.
    
    Args:
        d_model: Dimension of the model
        n_heads: Number of attention heads
        d_ff: Dimension of the feed-forward network
        dropout: Dropout probability
    """
    
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        
        # Layer normalization (GPT-3 uses pre-norm, applied before sub-layers)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            mask: Optional causal mask
        
        Returns:
            Output tensor of shape (batch_size, seq_len, d_model)
        """
        # Pre-norm architecture: LayerNorm before attention
        attn_output = self.attention(self.ln1(x), mask)
        x = x + self.dropout(attn_output)
        
        # Pre-norm architecture: LayerNorm before feed-forward
        ff_output = self.feed_forward(self.ln2(x))
        x = x + self.dropout(ff_output)
        
        return x


class GPT3(nn.Module):
    """
    GPT-3 Model: A large-scale autoregressive language model.
    
    The model consists of:
    1. Token embeddings
    2. Positional embeddings (learned)
    3. Stack of Transformer decoder blocks
    4. Final layer normalization
    5. Language modeling head (projects to vocabulary)
    
    Args:
        vocab_size: Size of the vocabulary
        max_seq_len: Maximum sequence length
        d_model: Dimension of the model (embedding dimension)
        n_layers: Number of Transformer blocks
        n_heads: Number of attention heads
        d_ff: Dimension of the feed-forward network
        dropout: Dropout probability
    """
    
    def __init__(self, vocab_size, max_seq_len=2048, d_model=768, 
                 n_layers=12, n_heads=12, d_ff=3072, dropout=0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.d_model = d_model
        
        # Token embeddings
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        
        # Positional embeddings (learned, not sinusoidal as in original Transformer)
        self.positional_embedding = nn.Embedding(max_seq_len, d_model)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Stack of Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        
        # Final layer normalization
        self.ln_f = nn.LayerNorm(d_model)
        
        # Language modeling head (no bias, weight tied with token embeddings)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        
        # Weight initialization
        self.apply(self._init_weights)
        
        # Tie weights between token embeddings and output layer (after initialization)
        # This reduces parameters and is standard practice in language models
        self.lm_head.weight = self.token_embedding.weight
        
    def _init_weights(self, module):
        """Initialize weights following GPT-3 paper recommendations."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
    
    def create_causal_mask(self, seq_len, device):
        """
        Create a causal (lower triangular) mask for autoregressive generation.
        This ensures that position i can only attend to positions <= i.
        
        Args:
            seq_len: Sequence length
            device: Device to create the mask on
        
        Returns:
            Mask tensor of shape (1, 1, seq_len, seq_len)
        """
        mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
        mask = mask.unsqueeze(0).unsqueeze(0)  # Add batch and head dimensions
        return mask
    
    def forward(self, input_ids, targets=None):
        """
        Forward pass of the GPT-3 model.
        
        Args:
            input_ids: Input token indices of shape (batch_size, seq_len)
            targets: Optional target token indices for computing loss (batch_size, seq_len)
        
        Returns:
            If targets is None: logits of shape (batch_size, seq_len, vocab_size)
            If targets is provided: (logits, loss)
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        # Create position indices
        positions = torch.arange(0, seq_len, dtype=torch.long, device=device)
        positions = positions.unsqueeze(0).expand(batch_size, seq_len)
        
        # Get embeddings
        token_emb = self.token_embedding(input_ids)  # (batch_size, seq_len, d_model)
        pos_emb = self.positional_embedding(positions)  # (batch_size, seq_len, d_model)
        
        # Combine embeddings
        x = self.dropout(token_emb + pos_emb)
        
        # Create causal mask
        mask = self.create_causal_mask(seq_len, device)
        
        # Pass through Transformer blocks
        for block in self.transformer_blocks:
            x = block(x, mask)
        
        # Final layer normalization
        x = self.ln_f(x)
        
        # Language modeling head
        logits = self.lm_head(x)  # (batch_size, seq_len, vocab_size)
        
        # Compute loss if targets are provided
        loss = None
        if targets is not None:
            # Flatten the logits and targets for cross-entropy loss
            # Note: Use ignore_index=-1 if your dataset marks padding tokens with -1
            loss = F.cross_entropy(
                logits.view(-1, self.vocab_size),
                targets.view(-1),
                ignore_index=-1  # Ignore padding tokens (if marked with -1)
            )
        
        if loss is not None:
            return logits, loss
        else:
            return logits
    
    @torch.no_grad()
    def generate(self, input_ids, max_new_tokens, temperature=1.0, top_k=None):
        """
        Generate new tokens autoregressively.
        
        Args:
            input_ids: Input token indices of shape (batch_size, seq_len)
            max_new_tokens: Number of new tokens to generate
            temperature: Sampling temperature (higher = more random)
            top_k: If set, only sample from top k most likely tokens
        
        Returns:
            Generated token indices of shape (batch_size, seq_len + max_new_tokens)
        """
        for _ in range(max_new_tokens):
            # Crop input_ids if it exceeds max_seq_len
            input_ids_cond = input_ids if input_ids.size(1) <= self.max_seq_len else input_ids[:, -self.max_seq_len:]
            
            # Forward pass
            logits = self.forward(input_ids_cond)
            
            # Get logits for the last position
            logits = logits[:, -1, :] / temperature
            
            # Optionally apply top-k filtering
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')
            
            # Apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)
            
            # Sample from the distribution
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Append to the sequence
            input_ids = torch.cat([input_ids, next_token], dim=1)
        
        return input_ids


def get_gpt3_config(model_size='small'):
    """
    Get GPT-3 model configurations based on the paper.
    
    GPT-3 Variants from the paper:
    - GPT-3 Small: 125M parameters
    - GPT-3 Medium: 350M parameters
    - GPT-3 Large: 760M parameters
    - GPT-3 XL: 1.3B parameters
    - GPT-3 2.7B: 2.7B parameters
    - GPT-3 6.7B: 6.7B parameters
    - GPT-3 13B: 13B parameters
    - GPT-3 175B: 175B parameters (full GPT-3)
    
    Args:
        model_size: Size of the model ('small', 'medium', 'large', 'xl', '2.7b', '6.7b', '13b', '175b')
    
    Returns:
        Dictionary with model configuration
    """
    configs = {
        'small': {
            'n_layers': 12,
            'd_model': 768,
            'n_heads': 12,
            'd_ff': 3072,
            'vocab_size': 50257,  # GPT-2/GPT-3 vocabulary size
            'max_seq_len': 2048,
            'dropout': 0.1
        },
        'medium': {
            'n_layers': 24,
            'd_model': 1024,
            'n_heads': 16,
            'd_ff': 4096,
            'vocab_size': 50257,
            'max_seq_len': 2048,
            'dropout': 0.1
        },
        'large': {
            'n_layers': 24,
            'd_model': 1536,
            'n_heads': 16,
            'd_ff': 6144,
            'vocab_size': 50257,
            'max_seq_len': 2048,
            'dropout': 0.1
        },
        'xl': {
            'n_layers': 24,
            'd_model': 2048,
            'n_heads': 32,
            'd_ff': 8192,
            'vocab_size': 50257,
            'max_seq_len': 2048,
            'dropout': 0.1
        },
        '2.7b': {
            'n_layers': 32,
            'd_model': 2560,
            'n_heads': 32,
            'd_ff': 10240,
            'vocab_size': 50257,
            'max_seq_len': 2048,
            'dropout': 0.1
        },
        '6.7b': {
            'n_layers': 32,
            'd_model': 4096,
            'n_heads': 32,
            'd_ff': 16384,
            'vocab_size': 50257,
            'max_seq_len': 2048,
            'dropout': 0.1
        },
        '13b': {
            'n_layers': 40,
            'd_model': 5120,
            'n_heads': 40,
            'd_ff': 20480,
            'vocab_size': 50257,
            'max_seq_len': 2048,
            'dropout': 0.1
        },
        '175b': {
            'n_layers': 96,
            'd_model': 12288,
            'n_heads': 96,
            'd_ff': 49152,
            'vocab_size': 50257,
            'max_seq_len': 2048,
            'dropout': 0.1
        }
    }
    
    if model_size not in configs:
        available = ', '.join(sorted(configs.keys()))
        raise ValueError(f"Unknown model size: {model_size}. Available sizes: {available}")
    
    return configs[model_size]


def count_parameters(model):
    """Count the number of trainable parameters in the model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# Example usage
if __name__ == "__main__":
    print("=" * 80)
    print("GPT-3 Implementation in PyTorch")
    print("Paper: Language Models are Few-Shot Learners (2020)")
    print("=" * 80)
    print()
    
    # Example 1: Create a small GPT-3 model
    print("Example 1: Creating a GPT-3 Small model")
    print("-" * 80)
    config = get_gpt3_config('small')
    model = GPT3(**config)
    print(f"Model created with {count_parameters(model):,} parameters")
    print(f"Configuration: {config}")
    print()
    
    # Example 2: Forward pass with dummy data
    print("Example 2: Forward pass with dummy input")
    print("-" * 80)
    batch_size = 2
    seq_len = 32
    
    # Create random input tokens
    input_ids = torch.randint(0, config['vocab_size'], (batch_size, seq_len))
    print(f"Input shape: {input_ids.shape}")
    
    # Forward pass
    logits = model(input_ids)
    print(f"Output logits shape: {logits.shape}")
    print()
    
    # Example 3: Generate tokens
    print("Example 3: Text generation (with random inputs)")
    print("-" * 80)
    # Start with a short sequence
    input_ids = torch.randint(0, config['vocab_size'], (1, 10))
    print(f"Starting sequence length: {input_ids.shape[1]}")
    
    # Generate 20 new tokens
    generated = model.generate(input_ids, max_new_tokens=20, temperature=0.8, top_k=40)
    print(f"Generated sequence length: {generated.shape[1]}")
    print()
    
    # Example 4: Show different model sizes
    print("Example 4: GPT-3 Model Variants")
    print("-" * 80)
    for size in ['small', 'medium', 'large', 'xl']:
        config = get_gpt3_config(size)
        model = GPT3(**config)
        params = count_parameters(model)
        print(f"GPT-3 {size.upper():8s}: {params:15,} parameters | "
              f"{config['n_layers']} layers | d_model={config['d_model']}")
    
    print()
    print("=" * 80)
    print("Implementation complete! The model is ready for training on your dataset.")
    print("=" * 80)

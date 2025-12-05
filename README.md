# paper-replications

A collection of clean, from-scratch implementations of influential machine learning papers.

## 📚 Implementations

### 1. Multi-Layer Perceptron (MLP) with Backpropagation (1986)

A clean, from-scratch implementation of the Multi-Layer Perceptron (MLP) algorithm, verifying the concepts presented in the seminal 1986 paper by Rumelhart, Hinton, and Williams.

This implementation relies **solely on NumPy**, avoiding high-level frameworks like PyTorch or TensorFlow to demonstrate the mathematical "first principles" of backpropagation.

**📄 Paper Details:**
- **Title:** Learning representations by back-propagating errors
- **Authors:** David E. Rumelhart, Geoffrey E. Hinton, Ronald J. Williams
- **Year:** 1986
- **Nature:** [Link to Paper](https://www.nature.com/articles/323533a0)

**📂 File:** `MLP_with_Backprop.ipynb`

---

### 2. GPT-3: Language Models are Few-Shot Learners (2020)

A comprehensive PyTorch implementation of the GPT-3 architecture from scratch. This implementation includes all key components of the transformer-based autoregressive language model:

- **Multi-Head Self-Attention mechanism** with causal masking
- **Transformer decoder blocks** with pre-layer normalization
- **Positional embeddings** (learned, not sinusoidal)
- **Feed-forward networks** with GELU activation
- **Text generation** with temperature and top-k sampling
- **Multiple model configurations** (Small, Medium, Large, XL, 2.7B, 6.7B, 13B, 175B)

**📄 Paper Details:**
- **Title:** Language Models are Few-Shot Learners
- **Authors:** Tom B. Brown et al. (OpenAI)
- **Year:** 2020
- **arXiv:** [Link to Paper](https://arxiv.org/abs/2005.14165)

**📂 File:** `gpt3_pytorch.py`

**🚀 Usage:**

```python
from gpt3_pytorch import GPT3, get_gpt3_config

# Create a GPT-3 Small model (125M parameters)
config = get_gpt3_config('small')
model = GPT3(**config)

# Forward pass
import torch
input_ids = torch.randint(0, config['vocab_size'], (2, 32))
logits = model(input_ids)

# Text generation
generated = model.generate(input_ids, max_new_tokens=50, temperature=0.8, top_k=40)
```

**📊 Model Variants:**

| Model Size | Parameters | Layers | d_model | n_heads |
|-----------|-----------|--------|---------|---------|
| Small     | 125M      | 12     | 768     | 12      |
| Medium    | 350M      | 24     | 1024    | 16      |
| Large     | 760M      | 24     | 1536    | 16      |
| XL        | 1.3B      | 24     | 2048    | 32      |
| 2.7B      | 2.7B      | 32     | 2560    | 32      |
| 6.7B      | 6.7B      | 32     | 4096    | 32      |
| 13B       | 13B       | 40     | 5120    | 40      |
| 175B      | 175B      | 96     | 12288   | 96      |

---

## 🛠️ Setup

Install required dependencies:

```bash
pip install -r requirements.txt
```

## 📝 License

This repository contains educational implementations of research papers for learning purposes.

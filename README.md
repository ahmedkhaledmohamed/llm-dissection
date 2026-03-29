# LLM Dissection

Interactive Jupyter notebooks that load **Qwen 2.5** and show you exactly what every component of a modern LLM does — before and after training.

## Why Qwen 2.5?

Qwen 2.5 uses every technique found in frontier models (GPT-5, Claude, Llama 4): Grouped Query Attention, Rotary Position Embeddings, SwiGLU, RMSNorm. It's Apache 2.0 licensed, has a 0.5B variant that runs on CPU, and is architecturally identical from 0.5B to 72B.

## Architecture

```
tokens → Embedding (W_E) → residual stream
                              │
    ┌─────────────────────────┤ × 24 layers (0.5B) / 28 layers (7B)
    │                         │
    │  RMSNorm → Attention (GQA + RoPE) → + residual
    │                         │
    │  RMSNorm → MLP (SwiGLU)            → + residual
    │                         │
    └─────────────────────────┘
                              │
           RMSNorm → Unembedding (W_U) → logits → probabilities
```

## Notebooks

| # | Notebook | What You'll Learn |
|---|----------|-------------------|
| 01 | [Architecture Overview](notebooks/01_architecture_overview.ipynb) | Every module, weight shapes, parameter budget, GQA/SwiGLU/RMSNorm/RoPE |
| 02 | [Before vs After Training](notebooks/02_before_vs_after_training.ipynb) | Random init vs trained: weight distributions, SVD spectra, embedding clusters |
| 03 | [Embeddings](notebooks/03_embeddings.ipynb) | Token similarity, nearest neighbors, UMAP visualization |
| 04 | [Attention Deep Dive](notebooks/04_attention_deep_dive.ipynb) | Per-head patterns, head taxonomy, GQA in action, semantic heads |
| 05 | [FFN as Memory](notebooks/05_ffn_as_memory.ipynb) | SwiGLU internals, gate sparsity, interpretable neurons |
| 06 | [Forward Pass Trace](notebooks/06_forward_pass_trace.ipynb) | Step-by-step through every layer with actual activations |
| 07 | [Logit Lens](notebooks/07_logit_lens.ipynb) | Watch predictions form layer by layer |
| 08 | [Residual Stream](notebooks/08_residual_stream.ipynb) | Norm growth, PCA trajectory, component decomposition |

## Setup

```bash
# Clone and install
git clone <repo-url> && cd llm-dissection
pip install -r requirements.txt

# Launch notebooks
jupyter notebook notebooks/
```

**Requirements**:
- Python 3.10+
- ~4GB RAM for 0.5B model (default, runs on CPU)
- ~16GB VRAM for 7B model (optional, set `MODEL_SIZE = "7b"`)

Model weights are downloaded automatically from HuggingFace on first run.

## Key Libraries

- [TransformerLens](https://github.com/TransformerLensOrg/TransformerLens) — hooks into every intermediate computation
- [HuggingFace Transformers](https://github.com/huggingface/transformers) — model loading and tokenization
- [PyTorch](https://pytorch.org/) — the foundation

## References

- [Qwen 2.5 Technical Report](https://arxiv.org/abs/2412.15115)
- [A Mathematical Framework for Transformer Circuits](https://transformer-circuits.pub/2021/framework/index.html) (Elhage et al., 2021)
- [The Logit Lens](https://www.lesswrong.com/posts/AcKRB8wDpdaN6v6ru/interpreting-gpt-the-logit-lens) (nostalgebraist, 2020)
- [Transformer Feed-Forward Layers Are Key-Value Memories](https://arxiv.org/abs/2012.14913) (Geva et al., 2021)
- [Sebastian Raschka's LLM Architecture Gallery](https://sebastianraschka.com/llm-architecture-gallery/)

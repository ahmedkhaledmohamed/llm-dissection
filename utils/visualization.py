import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
from utils.constants import COLORS


def plot_weight_histogram(tensor, title="Weight Distribution", bins=100, ax=None):
    """Histogram of weight values in a tensor."""
    data = tensor.detach().float().cpu().flatten().numpy()
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(data, bins=bins, alpha=0.7, edgecolor="none", color=COLORS["trained"])
    ax.set_title(title)
    ax.set_xlabel("Weight Value")
    ax.set_ylabel("Count")
    ax.axvline(0, color="black", linestyle="--", alpha=0.3)
    stats_text = f"μ={data.mean():.4f}, σ={data.std():.4f}"
    ax.text(0.98, 0.95, stats_text, transform=ax.transAxes, ha="right", va="top",
            fontsize=9, bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    return ax


def plot_weight_heatmap(tensor_2d, title="Weight Heatmap", max_size=256, ax=None):
    """2D heatmap of a weight matrix (subsampled if too large)."""
    data = tensor_2d.detach().float().cpu().numpy()
    if data.shape[0] > max_size or data.shape[1] > max_size:
        step_r = max(1, data.shape[0] // max_size)
        step_c = max(1, data.shape[1] // max_size)
        data = data[::step_r, ::step_c]
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    vmax = np.abs(data).max()
    ax.imshow(data, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_title(f"{title} ({tensor_2d.shape[0]}×{tensor_2d.shape[1]})")
    ax.set_xlabel("Columns")
    ax.set_ylabel("Rows")
    return ax


def plot_attention_pattern(pattern, tokens, title="Attention Pattern", ax=None):
    """Single-head attention pattern heatmap with token labels."""
    data = pattern.detach().float().cpu().numpy()
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(data, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(tokens)))
    ax.set_xticklabels(tokens, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(tokens)))
    ax.set_yticklabels(tokens, fontsize=8)
    ax.set_title(title)
    ax.set_xlabel("Key Position")
    ax.set_ylabel("Query Position")
    return ax


def plot_attention_grid(patterns, tokens, n_heads, title_prefix="Head", cols=4):
    """Grid of attention patterns for multiple heads."""
    rows = (n_heads + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    axes = np.array(axes).flatten()
    for h in range(n_heads):
        data = patterns[h].detach().float().cpu().numpy()
        axes[h].imshow(data, cmap="Blues", vmin=0, vmax=1)
        axes[h].set_title(f"{title_prefix} {h}", fontsize=9)
        axes[h].set_xticks([])
        axes[h].set_yticks([])
    for h in range(n_heads, len(axes)):
        axes[h].axis("off")
    fig.tight_layout()
    return fig


def plot_layer_norms(norms, title="Activation Norms Across Layers", ylabel="L2 Norm",
                     color=None, ax=None):
    """Bar chart of norms across layers."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(range(len(norms)), norms, color=color or COLORS["residual"], alpha=0.8)
    ax.set_title(title)
    ax.set_xlabel("Layer")
    ax.set_ylabel(ylabel)
    return ax


def plot_logit_lens(top_tokens_per_layer, confidences, tokens, ax=None):
    """Heatmap showing top-1 predicted token at each layer/position.

    Args:
        top_tokens_per_layer: list of lists of token strings [n_layers][n_positions]
        confidences: numpy array [n_layers, n_positions] of probabilities
        tokens: list of input token strings
    """
    n_layers, n_pos = confidences.shape
    if ax is None:
        fig, ax = plt.subplots(figsize=(max(10, n_pos * 1.2), max(8, n_layers * 0.4)))

    ax.imshow(confidences, cmap="YlOrRd", aspect="auto", vmin=0, vmax=1)

    for i in range(n_layers):
        for j in range(n_pos):
            token_text = top_tokens_per_layer[i][j]
            if len(token_text) > 8:
                token_text = token_text[:7] + "…"
            ax.text(j, i, token_text, ha="center", va="center", fontsize=7,
                    color="white" if confidences[i, j] > 0.5 else "black")

    ax.set_xticks(range(n_pos))
    ax.set_xticklabels(tokens, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(n_layers))
    ax.set_yticklabels([f"L{i}" for i in range(n_layers)], fontsize=8)
    ax.set_xlabel("Input Token Position")
    ax.set_ylabel("Layer")
    ax.set_title("Logit Lens: Top-1 Prediction at Each Layer")
    return ax


def plot_distribution_comparison(tensor_a, tensor_b, label_a="A", label_b="B",
                                  title="Distribution Comparison", bins=100, ax=None):
    """Overlay histograms for two tensors."""
    data_a = tensor_a.detach().float().cpu().flatten().numpy()
    data_b = tensor_b.detach().float().cpu().flatten().numpy()
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(data_a, bins=bins, alpha=0.6, label=label_a, color=COLORS["random"], edgecolor="none")
    ax.hist(data_b, bins=bins, alpha=0.6, label=label_b, color=COLORS["trained"], edgecolor="none")
    ax.set_title(title)
    ax.legend()
    ax.axvline(0, color="black", linestyle="--", alpha=0.3)
    return ax


def plot_svd_spectrum(tensor, title="Singular Value Spectrum", top_k=100, ax=None):
    """Plot singular values of a weight matrix."""
    data = tensor.detach().float().cpu()
    U, S, V = torch.linalg.svd(data, full_matrices=False)
    s_vals = S[:top_k].numpy()
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(s_vals, marker=".", markersize=3)
    ax.set_title(title)
    ax.set_xlabel("Singular Value Index")
    ax.set_ylabel("Singular Value")
    ax.set_yscale("log")
    return ax


def plot_cosine_similarity_matrix(tensors, labels=None, title="Cosine Similarity", ax=None):
    """Heatmap of pairwise cosine similarities between a list of vectors."""
    vecs = torch.stack([t.detach().float().cpu().flatten() for t in tensors])
    vecs_norm = vecs / vecs.norm(dim=1, keepdim=True)
    sim = (vecs_norm @ vecs_norm.T).numpy()
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 7))
    sns.heatmap(sim, ax=ax, cmap="RdBu_r", vmin=-1, vmax=1, square=True,
                xticklabels=labels or range(len(tensors)),
                yticklabels=labels or range(len(tensors)))
    ax.set_title(title)
    return ax

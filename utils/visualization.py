import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
from utils.constants import COLORS

# ── Global theme ──────────────────────────────────────────────────────────────
# Dark background, clean typography, no chartjunk

DARK_BG = "#1a1a2e"
DARK_SURFACE = "#16213e"
DARK_GRID = "#2a2a4a"
TEXT_COLOR = "#e0e0e0"
ACCENT_BLUE = "#4fc3f7"
ACCENT_ORANGE = "#ffb74d"
ACCENT_GREEN = "#81c784"
ACCENT_RED = "#ef5350"
ACCENT_PURPLE = "#ba68c8"
ACCENT_TEAL = "#4db6ac"

PALETTE = [ACCENT_BLUE, ACCENT_ORANGE, ACCENT_GREEN, ACCENT_RED, ACCENT_PURPLE, ACCENT_TEAL]


def apply_theme():
    """Apply the dark professional theme globally. Call once per notebook."""
    plt.rcParams.update({
        # Figure
        "figure.facecolor": DARK_BG,
        "figure.edgecolor": DARK_BG,
        "figure.dpi": 120,
        # Axes
        "axes.facecolor": DARK_SURFACE,
        "axes.edgecolor": DARK_GRID,
        "axes.labelcolor": TEXT_COLOR,
        "axes.titlecolor": TEXT_COLOR,
        "axes.grid": True,
        "axes.grid.axis": "both",
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "axes.labelsize": 10,
        "axes.prop_cycle": plt.cycler(color=PALETTE),
        # Grid
        "grid.color": DARK_GRID,
        "grid.alpha": 0.3,
        "grid.linewidth": 0.5,
        # Ticks
        "xtick.color": TEXT_COLOR,
        "ytick.color": TEXT_COLOR,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        # Text
        "text.color": TEXT_COLOR,
        "font.family": "sans-serif",
        "font.size": 10,
        # Legend
        "legend.facecolor": DARK_SURFACE,
        "legend.edgecolor": DARK_GRID,
        "legend.fontsize": 9,
        "legend.labelcolor": TEXT_COLOR,
        # Savefig
        "savefig.facecolor": DARK_BG,
        "savefig.edgecolor": DARK_BG,
    })


def _annotate_bar(ax, bars, fmt="{:.1f}", fontsize=8):
    """Add value labels on top of bar chart bars."""
    for bar in bars:
        h = bar.get_height()
        if h > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, h, fmt.format(h),
                    ha="center", va="bottom", fontsize=fontsize, color=TEXT_COLOR, alpha=0.8)


def _style_box(ax, text, x=0.98, y=0.95, ha="right", va="top"):
    """Add a floating stat box to an axis."""
    ax.text(x, y, text, transform=ax.transAxes, ha=ha, va=va, fontsize=9,
            color=TEXT_COLOR,
            bbox=dict(boxstyle="round,pad=0.4", facecolor=DARK_SURFACE,
                      edgecolor=DARK_GRID, alpha=0.9))


# ── Chart functions ───────────────────────────────────────────────────────────

def plot_weight_histogram(tensor, title="Weight Distribution", bins=100, ax=None, color=None):
    """Histogram of weight values with stats overlay."""
    data = tensor.detach().float().cpu().flatten().numpy()
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4))
    c = color or ACCENT_BLUE
    ax.hist(data, bins=bins, alpha=0.85, edgecolor="none", color=c, linewidth=0)
    ax.set_title(title)
    ax.set_xlabel("Weight Value")
    ax.set_ylabel("Count")
    ax.axvline(0, color=TEXT_COLOR, linestyle="--", alpha=0.25, linewidth=0.8)
    _style_box(ax, f"$\\mu$={data.mean():.4f}\n$\\sigma$={data.std():.4f}")
    return ax


def plot_weight_heatmap(tensor_2d, title="Weight Heatmap", max_size=256, ax=None):
    """2D heatmap of a weight matrix with diverging colormap."""
    data = tensor_2d.detach().float().cpu().numpy()
    if data.shape[0] > max_size or data.shape[1] > max_size:
        step_r = max(1, data.shape[0] // max_size)
        step_c = max(1, data.shape[1] // max_size)
        data = data[::step_r, ::step_c]
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    vmax = np.abs(data).max()
    im = ax.imshow(data, cmap="coolwarm", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_title(f"{title}  ({tensor_2d.shape[0]}x{tensor_2d.shape[1]})")
    ax.set_xlabel("Columns")
    ax.set_ylabel("Rows")
    plt.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    return ax


def plot_attention_pattern(pattern, tokens, title="Attention Pattern", ax=None):
    """Single-head attention heatmap with token labels and colorbar."""
    data = pattern.detach().float().cpu().numpy()
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(data, cmap="magma", vmin=0, vmax=data.max())
    ax.set_xticks(range(len(tokens)))
    ax.set_xticklabels(tokens, rotation=45, ha="right", fontsize=8, color=TEXT_COLOR)
    ax.set_yticks(range(len(tokens)))
    ax.set_yticklabels(tokens, fontsize=8, color=TEXT_COLOR)
    ax.set_title(title)
    ax.set_xlabel("Key Position")
    ax.set_ylabel("Query Position")
    plt.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    return ax


def plot_attention_grid(patterns, tokens, n_heads, title_prefix="Head", cols=4):
    """Grid of attention patterns using magma colormap."""
    rows = (n_heads + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(3.5 * cols, 3.5 * rows))
    axes = np.array(axes).flatten()
    for h in range(n_heads):
        data = patterns[h].detach().float().cpu().numpy()
        axes[h].imshow(data, cmap="magma", vmin=0, vmax=data.max())
        axes[h].set_title(f"{title_prefix} {h}", fontsize=9, color=TEXT_COLOR)
        axes[h].set_xticks([])
        axes[h].set_yticks([])
    for h in range(n_heads, len(axes)):
        axes[h].axis("off")
    fig.tight_layout()
    return fig


def plot_layer_norms(norms, title="Activation Norms Across Layers", ylabel="L2 Norm",
                     color=None, ax=None, style="bar"):
    """Bar or line chart of values across layers."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 4))
    c = color or ACCENT_TEAL
    if style == "line":
        ax.plot(range(len(norms)), norms, marker="o", markersize=5, color=c,
                linewidth=2, markeredgecolor="white", markeredgewidth=0.5)
    else:
        bars = ax.bar(range(len(norms)), norms, color=c, alpha=0.85, edgecolor="none")
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
        fig, ax = plt.subplots(figsize=(max(10, n_pos * 1.4), max(8, n_layers * 0.45)))

    im = ax.imshow(confidences, cmap="inferno", aspect="auto", vmin=0, vmax=1)

    for i in range(n_layers):
        for j in range(n_pos):
            token_text = top_tokens_per_layer[i][j]
            if len(token_text) > 8:
                token_text = token_text[:7] + "..."
            text_color = "white" if confidences[i, j] < 0.6 else "black"
            ax.text(j, i, token_text, ha="center", va="center", fontsize=7, color=text_color)

    ax.set_xticks(range(n_pos))
    ax.set_xticklabels(tokens, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(n_layers))
    ax.set_yticklabels([f"L{i}" for i in range(n_layers)], fontsize=8)
    ax.set_xlabel("Input Token Position")
    ax.set_ylabel("Layer")
    ax.set_title("Logit Lens: Top-1 Prediction at Each Layer")
    plt.colorbar(im, ax=ax, shrink=0.8, label="Confidence", pad=0.02)
    return ax


def plot_distribution_comparison(tensor_a, tensor_b, label_a="A", label_b="B",
                                  title="Distribution Comparison", bins=100, ax=None):
    """Overlay histograms with distinct colors and alpha."""
    data_a = tensor_a.detach().float().cpu().flatten().numpy()
    data_b = tensor_b.detach().float().cpu().flatten().numpy()
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(data_a, bins=bins, alpha=0.6, label=label_a, color=ACCENT_RED, edgecolor="none")
    ax.hist(data_b, bins=bins, alpha=0.6, label=label_b, color=ACCENT_BLUE, edgecolor="none")
    ax.set_title(title)
    ax.legend()
    ax.axvline(0, color=TEXT_COLOR, linestyle="--", alpha=0.25, linewidth=0.8)
    return ax


def plot_svd_spectrum(tensor, title="Singular Value Spectrum", top_k=100, ax=None, color=None):
    """Plot singular values of a weight matrix on log scale."""
    data = tensor.detach().float().cpu()
    S = torch.linalg.svdvals(data)[:top_k].numpy()
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4))
    c = color or ACCENT_BLUE
    ax.plot(S, marker=".", markersize=4, color=c, linewidth=1.5)
    ax.fill_between(range(len(S)), S, alpha=0.15, color=c)
    ax.set_title(title)
    ax.set_xlabel("Singular Value Index")
    ax.set_ylabel("Singular Value")
    ax.set_yscale("log")
    return ax


def plot_cosine_similarity_matrix(tensors, labels=None, title="Cosine Similarity", ax=None):
    """Heatmap of pairwise cosine similarities."""
    vecs = torch.stack([t.detach().float().cpu().flatten() for t in tensors])
    vecs_norm = vecs / vecs.norm(dim=1, keepdim=True)
    sim = (vecs_norm @ vecs_norm.T).numpy()
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 7))
    sns.heatmap(sim, ax=ax, cmap="coolwarm", vmin=-1, vmax=1, square=True,
                xticklabels=labels or range(len(tensors)),
                yticklabels=labels or range(len(tensors)),
                cbar_kws={"shrink": 0.8})
    ax.set_title(title)
    return ax


def plot_grouped_bars(x_labels, group_a, group_b, label_a, label_b,
                       color_a=None, color_b=None, title="", ylabel="", ax=None):
    """Side-by-side grouped bar chart."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(x_labels))
    width = 0.38
    ca = color_a or ACCENT_ORANGE
    cb = color_b or ACCENT_GREEN
    ax.bar(x - width / 2, group_a, width, label=label_a, color=ca, alpha=0.85, edgecolor="none")
    ax.bar(x + width / 2, group_b, width, label=label_b, color=cb, alpha=0.85, edgecolor="none")
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, fontsize=8)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.legend()
    return ax


def plot_pie(labels, values, colors=None, title="", ax=None):
    """Donut-style pie chart with clean labels."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 7))
    cs = colors or PALETTE[:len(values)]
    wedges, texts, autotexts = ax.pie(
        values, labels=labels, colors=cs, autopct="%1.1f%%",
        textprops={"fontsize": 10, "color": TEXT_COLOR},
        pctdistance=0.78, startangle=90,
        wedgeprops={"edgecolor": DARK_BG, "linewidth": 2}
    )
    for t in autotexts:
        t.set_fontsize(9)
        t.set_color(TEXT_COLOR)
    # Donut hole
    centre_circle = plt.Circle((0, 0), 0.55, fc=DARK_BG)
    ax.add_artist(centre_circle)
    ax.set_title(title, pad=20)
    return ax

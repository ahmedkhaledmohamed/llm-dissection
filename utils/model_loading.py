import torch
from utils.constants import TLENS_MODELS, HF_MODELS, DEFAULT_SIZE


def _get_device():
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _get_dtype(device):
    if device == "cpu":
        return torch.float32
    return torch.bfloat16


def load_tlens_model(size=DEFAULT_SIZE, device=None):
    """Load model via TransformerLens (HookedTransformer) for inspection."""
    from transformer_lens import HookedTransformer

    device = device or _get_device()
    dtype = _get_dtype(device)
    model_name = TLENS_MODELS[size]

    print(f"Loading {model_name} on {device} ({dtype})...")
    model = HookedTransformer.from_pretrained(
        model_name,
        device=device,
        dtype=dtype,
    )
    print(f"Loaded. Parameters: {sum(p.numel() for p in model.parameters()):,}")
    return model


def load_hf_model(size=DEFAULT_SIZE, device=None):
    """Load raw HuggingFace model (for before/after training comparison)."""
    from transformers import AutoModelForCausalLM

    device = device or _get_device()
    dtype = _get_dtype(device)
    model_name = HF_MODELS[size]

    print(f"Loading {model_name} (HuggingFace) on {device}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map=device if device != "mps" else None,
    )
    if device == "mps":
        model = model.to(device)
    print(f"Loaded. Parameters: {sum(p.numel() for p in model.parameters()):,}")
    return model


def load_random_model(size=DEFAULT_SIZE, device=None):
    """Load randomly initialized model with same architecture (no pretrained weights)."""
    from transformers import AutoModelForCausalLM, AutoConfig

    device = device or _get_device()
    dtype = _get_dtype(device)
    model_name = HF_MODELS[size]

    print(f"Creating random-init {model_name} on {device}...")
    config = AutoConfig.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_config(config).to(dtype=dtype, device=device)
    print(f"Created. Parameters: {sum(p.numel() for p in model.parameters()):,}")
    return model


def get_tokenizer(size=DEFAULT_SIZE):
    """Get the tokenizer for the model."""
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(HF_MODELS[size])


def format_param_count(n):
    """Format parameter count as human-readable string."""
    if n >= 1e9:
        return f"{n / 1e9:.1f}B"
    elif n >= 1e6:
        return f"{n / 1e6:.1f}M"
    elif n >= 1e3:
        return f"{n / 1e3:.1f}K"
    return str(n)

# TransformerLens model names
TLENS_MODELS = {
    "0.5b": "qwen2.5-0.5b",
    "7b": "qwen2.5-7b",
}

# HuggingFace model names
HF_MODELS = {
    "0.5b": "Qwen/Qwen2.5-0.5B",
    "7b": "Qwen/Qwen2.5-7B",
}

DEFAULT_SIZE = "0.5b"

EXAMPLE_PROMPTS = [
    "The capital of France is",
    "In a distant galaxy, the last star began to",
    "def fibonacci(n):\n    if n <= 1:\n        return n\n    return",
    "The quick brown fox jumps over the",
    "Water freezes at a temperature of",
]

# Consistent colors across notebooks
COLORS = {
    "embedding": "#4C72B0",
    "attention": "#DD8452",
    "mlp": "#55A868",
    "norm": "#C44E52",
    "unembedding": "#8172B3",
    "residual": "#937860",
    "trained": "#4C72B0",
    "random": "#C44E52",
}

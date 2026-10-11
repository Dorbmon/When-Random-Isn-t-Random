"""Architecture helpers so the circuit scripts run on Qwen3 and on Gemma-style models.

Gemma-style models load as multimodal wrappers: their text layers, final norm and config sit
under `language_model`, and some layers use sliding-window attention. Every custom attention
mask must respect a layer's window, or even the "no knockout" mask changes the computation.
"""


def text_parts(model):
    """Return (layers, final_norm, lm_head, text_config)."""
    cand = [model, getattr(model, "model", None)]
    for c in list(cand):
        if c is not None:
            cand.append(getattr(c, "language_model", None))
    for c in cand:
        if c is None:
            continue
        layers = getattr(c, "layers", None)
        if layers is None and hasattr(c, "model"):
            layers = getattr(c.model, "layers", None)
            c = c.model if layers is not None else c
        if layers is not None:
            norm = getattr(c, "norm", None)
            cfg = getattr(model.config, "text_config", None) or model.config
            return layers, norm, model.lm_head if hasattr(model, "lm_head") else model.get_output_embeddings(), cfg
    raise RuntimeError("could not locate decoder layers")


def layer_windows(cfg, n_layers):
    """Per-layer sliding window (None = full attention)."""
    types = getattr(cfg, "layer_types", None)
    win = getattr(cfg, "sliding_window", None)
    out = []
    for i in range(n_layers):
        t = types[i] if types else "full_attention"
        out.append(win if (t and "sliding" in t and win) else None)
    return out


def base_mask(T, window, device):
    """Boolean (T, T) causal mask, restricted to a sliding window if one is given."""
    import torch
    m = torch.tril(torch.ones(T, T, dtype=torch.bool, device=device))
    if window:
        idx = torch.arange(T, device=device)
        m &= (idx[:, None] - idx[None, :]) < window
    return m

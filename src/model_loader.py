"""
model_loader.py — the ONLY module that knows about:
  - file extensions (.keras vs .weights.h5)
  - board encoding types (A, B, B_flat) and their perspective-flip rules
  - which custom Keras layers each group's models require

Everything else in the codebase calls predict_probs() or encode_board() and
stays completely encoding-agnostic.

Encoding reference (determined by inspecting each group's backend code):
  Type "B"      — (6, 7, 2) float32, two-channel, perspective-flipped.
                  ch0 = current player's pieces, ch1 = opponent's pieces.
                  Used by: Stiles CNN, Stiles Transformer (M1), Zan CNN.
  Type "A"      — (6, 7, 1) float32, single signed channel.
                  +1 = current player, -1 = opponent (board negated for player -1).
                  Used by: Luke CNN, Luke Transformer.
  Type "B_flat" — (42, 2) float32, Type-B but board flattened row-major.
                  Used by: Zan Transformer (weights-only file, flat input arch).
"""
import importlib.util
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import tensorflow as tf

from . import config as _cfg


# ── Helpers ───────────────────────────────────────────────────────────────────

def _import_from_path(module_name: str, file_path: Path):
    """Import a .py file as a module given its absolute path."""
    spec = importlib.util.spec_from_file_location(module_name, str(file_path))
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ── ModelWrapper ──────────────────────────────────────────────────────────────

@dataclass
class ModelWrapper:
    """
    Pairs a Keras model with its encoding convention and a display name.

    encoding : "B"      → (6,7,2) two-channel perspective-flipped
               "A"      → (6,7,1) single-channel, negated for player -1
               "B_flat" → (42,2)  flattened two-channel perspective-flipped
    """
    model:    Any   # tf.keras.Model
    encoding: str   # "A", "B", or "B_flat"
    name:     str   # human-readable label


# ── Board encoding ────────────────────────────────────────────────────────────

def encode_board(wrapper: ModelWrapper, board: np.ndarray, player: int) -> np.ndarray:
    """
    Convert a canonical board (6×7 int8, +1/−1/0) to the numpy array the
    underlying model expects, WITHOUT a batch dimension.

    The encoding always places the current `player`'s pieces in channel 0
    (or as +1 in Type A) so the model always sees its own pieces the same way
    regardless of which side it is playing.
    """
    enc = wrapper.encoding

    if enc == "B":
        # Two-channel (6, 7, 2): ch0 = my pieces, ch1 = opponent's pieces
        if player == +1:
            ch0 = (board == +1).astype(np.float32)
            ch1 = (board == -1).astype(np.float32)
        else:
            ch0 = (board == -1).astype(np.float32)
            ch1 = (board == +1).astype(np.float32)
        return np.stack([ch0, ch1], axis=-1)          # shape (6, 7, 2)

    elif enc == "A":
        # Single-channel (6, 7, 1): negate board when playing as -1 so the
        # model always sees itself as the +1 player
        b = board.astype(np.float32)
        if player == -1:
            b = -b
        return b[..., np.newaxis]                     # shape (6, 7, 1)

    elif enc == "B_flat":
        # Flattened two-channel (42, 2): same perspective logic as "B"
        flat = board.flatten().astype(np.float32)
        if player == +1:
            ch0 = (flat == +1).astype(np.float32)
            ch1 = (flat == -1).astype(np.float32)
        else:
            ch0 = (flat == -1).astype(np.float32)
            ch1 = (flat == +1).astype(np.float32)
        return np.stack([ch0, ch1], axis=-1)          # shape (42, 2)

    else:
        raise ValueError(f"Unknown encoding type: '{enc}'")


def predict_probs(wrapper: ModelWrapper, board: np.ndarray, player: int) -> np.ndarray:
    """
    Run inference and return a (7,) float32 array of raw softmax probabilities
    over all 7 columns.  Illegal-move masking is the caller's responsibility.

    Uses the model's direct `__call__` instead of `.predict()`. For single-sample
    inference (which is what self-play does) `.predict()` has per-call Python
    overhead (progress bars, callbacks, graph re-tracing) that dominates the
    actual compute — the direct call is typically 3–5x faster on CPU and
    materially faster on GPU too.
    """
    # Add batch dimension
    x = encode_board(wrapper, board, player)[np.newaxis, ...].astype(np.float32)
    raw = wrapper.model(x, training=False)

    # Handle single-output models vs dual-output (list [policy, value])
    if isinstance(raw, (list, tuple)):
        probs = raw[0].numpy().flatten()     # take policy head; flatten batch dim
    else:
        probs = raw.numpy().flatten()        # single output; flatten batch dim

    return probs[:7].astype(np.float32)


# ── Model loading ─────────────────────────────────────────────────────────────

def load_all_models() -> dict:
    """
    Load M1 and all initial M2 candidates.  Returns a dict mapping name → ModelWrapper.

    Key "m1" is the Stiles Transformer that will be trained.
    All other keys are initial M2 opponents (never removed from the pool).
    Loading the Zan Transformer requires rebuilding its architecture from
    model_wrappers.py and then calling load_weights().
    """
    root = _cfg.ROOT
    models = {}

    # ── Stiles models — Type B, standard .keras load ──────────────────────────
    print("Loading Stiles models…")
    m1_keras = tf.keras.models.load_model(str(_cfg.M1_PATH), compile=False)
    models["m1"] = ModelWrapper(m1_keras, "B", "Stiles Transformer (M1)")

    # Load the original transformer again as a separate model object for the pool.
    # This copy stays frozen; M1's copy is the one trained.
    stiles_transformer_orig = tf.keras.models.load_model(
        str(_cfg.M2_PATHS["stiles_transformer_orig"]), compile=False
    )
    models["stiles_transformer_orig"] = ModelWrapper(
        stiles_transformer_orig, "B", "Stiles Transformer (original)"
    )

    stiles_cnn = tf.keras.models.load_model(
        str(_cfg.M2_PATHS["stiles_cnn"]), compile=False
    )
    models["stiles_cnn"] = ModelWrapper(stiles_cnn, "B", "Stiles CNN")

    # ── Luke models — Type A, custom layers defined in inference.py ───────────
    print("Loading Luke models…")
    luke_mod = _import_from_path(
        "luke_inference",
        root / "Luke Group Models" / "inference.py"
    )
    luke_custom = {
        "AddPositionEmb": luke_mod.AddPositionEmb,
        "ClassToken":     luke_mod.ClassToken,
    }

    # Luke CNN uses no custom layers
    luke_cnn = tf.keras.models.load_model(
        str(_cfg.M2_PATHS["luke_cnn"]), compile=False
    )
    models["luke_cnn"] = ModelWrapper(luke_cnn, "A", "Luke CNN")

    # Luke Transformer requires AddPositionEmb and ClassToken custom objects
    luke_transformer = tf.keras.models.load_model(
        str(_cfg.M2_PATHS["luke_transformer"]),
        custom_objects=luke_custom,
        compile=False,
    )
    models["luke_transformer"] = ModelWrapper(luke_transformer, "A", "Luke Transformer")

    # ── Zan CNN — Type B, standard .keras load, no custom layers ─────────────
    # The 226 MB Zan CNN file is gitignored, so it may be missing on Colab or
    # on a teammate's machine that hasn't downloaded it. Skip gracefully.
    print("Loading Zan models…")
    zan_cnn_path = _cfg.M2_PATHS.get("zan_cnn")
    if zan_cnn_path is not None and zan_cnn_path.exists():
        zan_cnn = tf.keras.models.load_model(str(zan_cnn_path), compile=False)
        models["zan_cnn"] = ModelWrapper(zan_cnn, "B", "Zan CNN")
    else:
        print("  (skipping Zan CNN — not in M2_PATHS or file missing)")

    # ── Zan Transformer — Type B_flat, weights-only file ─────────────────────
    # The .weights.h5 file stores only weights, not the architecture.
    # We must rebuild the exact architecture from model_wrappers.py first.
    zan_mod = _import_from_path(
        "zan_wrappers",
        root / "Zan Group Models" / "model_wrappers.py"
    )
    zan_transformer = zan_mod.build_connect4_transformer(
        hidden_dim=128, num_layers=6, num_heads=8, mlp_dim=256, dropout_rate=0.1
    )
    zan_transformer.load_weights(str(_cfg.M2_PATHS["zan_transformer"]))
    models["zan_transformer"] = ModelWrapper(
        zan_transformer, "B_flat", "Zan Transformer"
    )

    print(f"\nLoaded {len(models)} models:")
    for name, w in models.items():
        print(f"  {name:30s}  encoding={w.encoding}  ({w.name})")
    return models

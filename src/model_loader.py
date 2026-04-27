"""
model_loader.py — the ONLY module that knows about:
  - file extensions (.keras vs .h5 vs .weights.h5)
  - board encoding types (A, B, B_flat) and their perspective-flip rules
  - which custom Keras layers each group's models require
  - where large models live (in-repo, per-user cache, GitHub Release)

Everything else in the codebase calls predict_probs() or encode_board() and
stays completely encoding-agnostic.

Encoding reference (determined by inspecting each group's backend code):
  Type "B"      — (6, 7, 2) float32, two-channel, perspective-flipped.
                  ch0 = current player's pieces, ch1 = opponent's pieces.
                  Used by: Stiles CNN, Stiles Transformer (M1), Zan CNN,
                  the trained SAC/AC/DQN models.
  Type "A"      — (6, 7, 1) float32, single signed channel.
                  +1 = current player, -1 = opponent (board negated for player -1).
                  Used by: Luke CNN, Luke Transformer.
  Type "B_flat" — (42, 2) float32, Type-B but board flattened row-major.
                  Used by: Zan Transformer (weights-only file, flat input arch).
"""
import importlib.util
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import tensorflow as tf

from . import config as _cfg


# ── Silence one specific, benign Keras 3 warning ─────────────────────────────
# Every model() call emits a UserWarning about the input "structure" because
# we pass a positional tensor instead of {'input_layer': tensor}. The
# behaviour is identical either way; the warning just clutters the eval
# output once per inference call (i.e. thousands of times per round-robin).
# We filter it narrowly by message so genuine Keras warnings still surface.
warnings.filterwarnings(
    "ignore",
    message=r"The structure of `inputs` doesn't match the expected structure.*",
    category=UserWarning,
)


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
    Run inference and return a (7,) float32 array of raw softmax/Q scores
    over all 7 columns. Illegal-move masking is the caller's responsibility.

    Uses the model's direct `__call__` instead of `.predict()`. For single-sample
    inference (which is what self-play does) `.predict()` has per-call Python
    overhead (progress bars, callbacks, graph re-tracing) that dominates the
    actual compute — the direct call is typically 3–5x faster on CPU and
    materially faster on GPU too.

    Handles three output shapes:
      - single output (B, 7)                    → plain policy network
      - list/tuple [policy (B, 7), ...]         → dual-head (AC, SAC)
      - list/tuple [..., q_values (B, 7)]       → DQN-style networks

    The FIRST output whose last dim is 7 wins. This matches how SAC models
    are built (policy first, then Q-values).
    """
    # Add batch dimension
    x = encode_board(wrapper, board, player)[np.newaxis, ...].astype(np.float32)
    raw = wrapper.model(x, training=False)

    # Unwrap list/tuple output, prefer 7-column outputs
    if isinstance(raw, (list, tuple)):
        picked = None
        for o in raw:
            try:
                shape = o.shape
                if len(shape) >= 1 and shape[-1] == 7:
                    picked = o
                    break
            except Exception:
                continue
        raw = picked if picked is not None else raw[0]

    probs = raw.numpy().flatten() if hasattr(raw, "numpy") else np.asarray(raw).flatten()
    return probs[:7].astype(np.float32)


# ── Model loading ─────────────────────────────────────────────────────────────

# The Zan CNN (final_supervised_256f.keras, 226 MB) exceeds GitHub's 100 MB
# git-push limit, so it is hosted as a GitHub Release asset rather than
# committed to the repo. Every machine (your Mac, Colab, a teammate's laptop)
# looks in three places, in order:
#   1. The canonical in-repo path  (Zan Group Models/final_supervised_256f.keras)
#   2. A per-user cache            (~/.keras/connect4_rl_arena/...)
#   3. The GitHub Release asset    (downloaded to the cache, reused thereafter)
# The file is static for the project, so download-once-and-cache is safe.

ZAN_CNN_RELEASE_URL = (
    "https://github.com/Stiles-Clements1/connect4-rl-arena/"
    "releases/download/models-v1/final_supervised_256f.keras"
)
ZAN_CNN_CACHE_DIR      = Path.home() / ".keras" / "connect4_rl_arena"
# The real file is ~226 MB. Anything dramatically smaller is a truncated
# download (interrupted network, Ctrl+C mid-fetch, etc.) and Keras will
# fail to open it. Validate on every run so a bad cache self-heals.
ZAN_CNN_MIN_VALID_SIZE = 200 * 1024 * 1024


def _resolve_zan_cnn_path() -> Optional[Path]:
    """
    Return a local filesystem path to the Zan CNN file, or None if it cannot
    be located or fetched. Checks, in order:
      1. the in-repo path from config.py
      2. a per-user cache under ~/.keras/connect4_rl_arena/ — only if the
         cached file passes a size check. Partial downloads are deleted
         and retried instead of being silently handed to Keras.
      3. downloads from the GitHub Release into a .partial temp path first,
         then renames to the final path only on complete success. This
         ensures an interrupted download can never masquerade as a valid
         cache on the next run.
    """
    # 1. In-repo location (your Mac, or anyone who dropped the file in manually)
    local = _cfg.M2_PATHS.get("zan_cnn")
    if local is not None and local.exists():
        return local

    # 2. Per-user cache. Reject if size looks like a partial download.
    ZAN_CNN_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cached = ZAN_CNN_CACHE_DIR / "final_supervised_256f.keras"
    if cached.exists():
        size = cached.stat().st_size
        if size >= ZAN_CNN_MIN_VALID_SIZE:
            return cached
        print(f"  Cached Zan CNN at {cached} is only {size / 1024**2:.0f} MB "
              f"(need ~{ZAN_CNN_MIN_VALID_SIZE / 1024**2:.0f}+ MB) — partial "
              f"download, removing and re-fetching.")
        cached.unlink()

    # 3. Download into a temp .partial path, verify size, then rename.
    tmp = cached.with_suffix(".keras.partial")
    if tmp.exists():
        tmp.unlink()
    try:
        import urllib.request
        print("  Fetching Zan CNN from the GitHub Release (one-time, ~226 MB)…")
        urllib.request.urlretrieve(ZAN_CNN_RELEASE_URL, tmp)
        if tmp.stat().st_size < ZAN_CNN_MIN_VALID_SIZE:
            raise IOError(
                f"downloaded file is only {tmp.stat().st_size / 1024**2:.0f} MB, "
                f"expected ~226 MB — truncated response"
            )
        tmp.rename(cached)
        return cached
    except Exception as exc:
        # Clean up partial / incomplete files so the next run starts fresh
        for p in (tmp, cached):
            if p.exists():
                p.unlink()
        print(f"  (could not fetch Zan CNN from {ZAN_CNN_RELEASE_URL}: {exc})")
        return None


def load_all_models() -> dict:
    """
    Load M1 and all initial M2 candidates from the explicit paths declared
    in config.M1_PATH / config.M2_PATHS. Returns a dict mapping name →
    ModelWrapper.

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

    # ── Zan CNN — Type B; 226 MB file hosted on a GitHub Release ─────────────
    print("Loading Zan models…")
    zan_cnn_path = _resolve_zan_cnn_path()
    if zan_cnn_path is not None:
        zan_cnn = tf.keras.models.load_model(str(zan_cnn_path), compile=False)
        models["zan_cnn"] = ModelWrapper(zan_cnn, "B", "Zan CNN")
        print(f"  Zan CNN loaded from {zan_cnn_path}")
    else:
        print("  (skipping Zan CNN — not available locally or via Release)")

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

    # Hard structural check — declared encoding must match the model's
    # real input shape. Raises immediately on any mismatch.
    for w in models.values():
        verify_encoding(w)

    return models


# ─────────────────────────────────────────────────────────────────────────────
# Auto-discovery: scan the whole repo for .keras / .h5 files not explicitly
# configured. Picks up trained SAC, AC, DQN, PG snapshots without you having
# to wire each one into config.py.
# ─────────────────────────────────────────────────────────────────────────────

# File suffixes the discovery scan will consider. We look for everything a
# Keras model might live in: the modern .keras zip format and the two
# legacy HDF5 variants (full-model .h5 and weights-only .weights.h5).
_DISCOVER_SUFFIXES = (".keras", ".h5")

# Substrings in filename (case-insensitive) that exclude a file from
# discovery. These are byproducts of training we don't want as opponents:
#   - "target"   : SAC/AC target Q networks (not policy networks)
#   - "snapshot" : in-pool self-copies from mid-training
#   - ".partial" : interrupted downloads (resolver cleans these up too)
# Note: the `checkpoints/` directory is excluded at the path level by
# skip_dir_fragments below, so training checkpoints never even reach
# these filename checks.
_DISCOVER_EXCLUDE_PATTERNS_DEFAULT = ("target", "snapshot", ".partial")


def _infer_encoding_from_shape(input_shape) -> Optional[str]:
    """
    Map a Keras input shape to one of "A", "B", "B_flat", or None.

    Keras input shapes typically include a leading batch dim (None), but
    different backends and TensorFlow versions present this differently
    (`TensorShape([None, 6, 7, 2])`, `(None, 6, 7, 2)`, or occasionally a
    concrete batch like `(32, 6, 7, 2)`). To be robust, we match against
    the TRAILING dims — the last 3 (or 2 for flat) of the shape — regardless
    of what the batch dim looks like.
    """
    try:
        dims = tuple(input_shape)
    except Exception:
        return None
    if not dims:
        return None

    # Compare the trailing dims against each known encoding's shape.
    last3 = dims[-3:] if len(dims) >= 3 else None
    last2 = dims[-2:] if len(dims) >= 2 else None

    if last3 == (6, 7, 2):
        return "B"
    if last3 == (6, 7, 1):
        return "A"
    if last2 == (6, 7):        # rare: single channel omitted
        return "A"
    if last2 == (42, 2):
        return "B_flat"
    return None


def _try_load_generic(path: Path, luke_custom: Optional[dict]) -> Optional[tf.keras.Model]:
    """
    Try to load a full Keras model from `path`. First attempt uses Luke's
    custom layers (covers Luke Transformer siblings); second attempt with no
    custom objects (covers everything else). Returns the model or None.
    """
    for co in (luke_custom, None):
        try:
            return tf.keras.models.load_model(str(path), custom_objects=co, compile=False)
        except Exception:
            continue
    return None


def discover_extra_models(
    search_dirs: Optional[list] = None,
    exclude_name_patterns: Optional[tuple] = None,
    verbose: bool = True,
) -> dict:
    """
    Recursively scan folders for `.keras` / `.h5` files that are NOT already
    loaded by load_all_models(), load each, auto-detect its encoding from
    input shape, and return a dict of {filename_stem: ModelWrapper}.

    Intended for picking up FINISHED, promoted trained models without
    having to manually wire each one into config.py. Drop a model file
    into `RL models/` (the canonical home for tournament-ready agents) or
    any of the `*Group Models/` folders and it will show up as an
    opponent on the next notebook run. The `checkpoints/` folder is
    DELIBERATELY excluded — those are in-training snapshots, not
    deployable agents, and should not appear in evaluation.

    Parameters
    ----------
    search_dirs : list of Path, optional
        Directories to scan. Defaults to the entire repo root — this is by
        design so teammates can drop a model anywhere in the repo and it
        gets picked up. Pass a narrower list if you want to constrain.
    exclude_name_patterns : tuple of str, optional
        Case-insensitive substrings that exclude a file from discovery.
        Defaults to ("target", "snapshot", ".partial").
    verbose : bool
        Print a line per discovered/skipped file.

    Returns
    -------
    dict[str, ModelWrapper]
        Keyed by filename stem (e.g. "sac_zan", "enhanced_dqn_optimized").
        Models already registered via config.M1_PATH / config.M2_PATHS
        are skipped to avoid duplicates.
    """
    root = _cfg.ROOT
    if search_dirs is None:
        # Scan the entire repo root by default. Subfolders git doesn't track
        # (venvs, caches, etc.) are filtered below.
        search_dirs = [root]
    if exclude_name_patterns is None:
        exclude_name_patterns = _DISCOVER_EXCLUDE_PATTERNS_DEFAULT

    # Folders to skip wholesale inside the scan — paths containing any of
    # these substrings (case-insensitive) are pruned. Keeps us out of
    # virtualenvs, per-user caches, git metadata, notebook metadata, and
    # in-training checkpoint directories. Only FINISHED, promoted models
    # (canonically in `RL models/` or the `*Group Models/` folders) should
    # appear as evaluation opponents; mid-training snapshots under
    # `checkpoints/` are training state, not tournament agents.
    skip_dir_fragments = (
        ".git", "__pycache__", ".ipynb_checkpoints", "wandb",
        "venv", ".venv", "node_modules",
        "checkpoints",
    )

    # Resolve the paths that load_all_models handles — skip them here.
    known_paths = set()
    try:
        known_paths.add(str(Path(_cfg.M1_PATH).resolve()))
        for p in _cfg.M2_PATHS.values():
            known_paths.add(str(Path(p).resolve()))
    except Exception:
        pass

    # Luke's custom layers — needed to load models trained by Luke's team
    # (AddPositionEmb, ClassToken). Import once, reuse per file.
    luke_custom = None
    try:
        luke_mod = _import_from_path(
            "luke_inference",
            root / "Luke Group Models" / "inference.py",
        )
        luke_custom = {
            "AddPositionEmb": luke_mod.AddPositionEmb,
            "ClassToken":     luke_mod.ClassToken,
        }
    except Exception:
        luke_custom = None

    discovered = {}
    if verbose:
        print("Discovering extra model files (.keras / .h5)…")

    candidates = []
    for d in search_dirs:
        if not Path(d).exists():
            continue
        for suffix in _DISCOVER_SUFFIXES:
            for path in Path(d).rglob(f"*{suffix}"):
                # Prune skip-dirs anywhere in the path
                lowered = str(path).lower()
                if any(frag in lowered for frag in skip_dir_fragments):
                    continue
                candidates.append(path)

    # Deduplicate + stable order so logs are readable
    candidates = sorted(set(candidates))

    for path in candidates:
        resolved = str(path.resolve())
        if resolved in known_paths:
            continue
        name_l = path.name.lower()
        if any(pat in name_l for pat in exclude_name_patterns):
            continue
        # .weights.h5 files are weights-only — they need architecture rebuild
        # code that we don't have generically. Skip with a note.
        if name_l.endswith(".weights.h5"):
            if verbose:
                print(f"  (skipped {path.name} — weights-only .h5 needs architecture rebuild)")
            continue

        model = _try_load_generic(path, luke_custom)
        if model is None:
            if verbose:
                print(f"  (skipped {path.name} — could not load as a full Keras model)")
            continue

        # Determine encoding from input shape
        input_shape = None
        try:
            input_shape = model.input.shape
        except Exception:
            try:
                input_shape = model.inputs[0].shape
            except Exception:
                pass
        encoding = _infer_encoding_from_shape(input_shape) if input_shape is not None else None
        if encoding is None:
            if verbose:
                print(f"  (skipped {path.name} — unrecognized input shape {input_shape})")
            continue

        name = path.stem
        # Strip the extra ".weights" suffix if someone passes a
        # full-model .weights.h5 (rare — wouldn't normally reach here).
        if name.endswith(".weights"):
            name = name[: -len(".weights")]
        if name in discovered:
            if verbose:
                print(f"  (skipped {path} — duplicate stem '{name}' already discovered)")
            continue

        discovered[name] = ModelWrapper(model, encoding, name)
        if verbose:
            try:
                rel = path.relative_to(root)
            except ValueError:
                rel = path
            print(f"  + {name:32s}  encoding={encoding}  ({rel})")

    if not discovered and verbose:
        print("  (no extra models discovered)")
    return discovered


def load_all_models_with_discovery(
    search_dirs: Optional[list] = None,
    verbose: bool = True,
) -> dict:
    """
    Convenience wrapper: load_all_models() + discover_extra_models().
    The configured group models always take precedence over discovered
    duplicates (by stem name). Pass `search_dirs` to narrow the scan.
    """
    models = load_all_models()
    extras = discover_extra_models(search_dirs=search_dirs, verbose=verbose)
    for name, wrapper in extras.items():
        if name not in models:
            models[name] = wrapper
    return models


# ─────────────────────────────────────────────────────────────────────────────
# Encoding verification
# ─────────────────────────────────────────────────────────────────────────────
#
# Two ways an encoding label can be wrong:
#   (1) STRUCTURAL — the wrapper says "B" but the Keras model expects (6,7,1).
#       Keras itself errors out on the first call; easy to catch but only
#       AFTER you've spent a minute warming up the round-robin. The static
#       check below catches this at load time.
#   (2) SEMANTIC — wrapper says "B" and the model expects (6,7,2), but
#       channel 0 was trained as "opponent's pieces" and our encode_board()
#       puts current-player there. Same shape, opposite meaning. Keras
#       can't see this; the model just plays badly. The vs-random
#       behavioural check (in the notebook) catches it.
#
# The static check is wired into load_all_models / discover so a label
# that doesn't match the model's input is impossible to ship silently.

# Expected trailing dims of model.input.shape per encoding label.
_EXPECTED_TRAILING = {
    "A":      ((6, 7, 1), (6, 7)),
    "B":      ((6, 7, 2),),
    "B_flat": ((42, 2),),
}


def verify_encoding(wrapper: ModelWrapper) -> tuple:
    """
    Confirm the wrapper's declared encoding matches its underlying model's
    input shape. Returns (declared_encoding, input_shape, "OK" | reason).
    Raises ValueError on a structural mismatch — silent encoding errors
    have a long history of corrupting evaluation results, so we fail loud.
    """
    try:
        shape = tuple(wrapper.model.input.shape)
    except Exception:
        try:
            shape = tuple(wrapper.model.inputs[0].shape)
        except Exception:
            shape = None
    if shape is None:
        return wrapper.encoding, None, "could not introspect input shape"

    trailing = shape[1:]   # drop the batch dim (None / concrete int)
    options = _EXPECTED_TRAILING.get(wrapper.encoding)
    if options is None:
        raise ValueError(
            f"{wrapper.name}: unknown encoding label '{wrapper.encoding}'"
        )
    if trailing not in options:
        raise ValueError(
            f"{wrapper.name}: declared encoding {wrapper.encoding!r} expects "
            f"trailing dims in {options}, but the model's actual input has "
            f"trailing dims {trailing}. This will produce wrong predictions "
            f"silently or crash on first call. Fix the encoding label in "
            f"model_loader.load_all_models or check the discovery shape "
            f"inference for this file."
        )
    return wrapper.encoding, shape, "OK"


def verify_all_encodings(models: dict, verbose: bool = True) -> dict:
    """
    Run verify_encoding on every wrapper in `models`. Prints a one-line
    summary per model. Raises on the first mismatch (via verify_encoding).
    Returns {name: (encoding, input_shape, status)}.
    """
    results = {}
    if verbose:
        print(f"\n{'Model':32s} {'Enc':8s} {'Input shape':22s} {'Status':30s}")
        print("─" * 96)
    for name, w in models.items():
        enc, shape, status = verify_encoding(w)
        results[name] = (enc, shape, status)
        if verbose:
            print(f"{name:32s} {enc:8s} {str(shape):22s} {status:30s}")
    if verbose:
        print("─" * 96)
        print(f"All {len(models)} encodings verified against actual model input shapes.")
    return results

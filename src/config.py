"""
config.py — single source of truth for all hyperparameters, paths, and seeds.

To change any training setting, edit ONLY this file.
"""
from pathlib import Path

# Project root is one directory above this file (Opti Proj 3/)
ROOT = Path(__file__).resolve().parent.parent

# ── Model paths ───────────────────────────────────────────────────────────────
# M1: the Stiles Transformer we improve via policy gradient (Questions 1–3)
M1_PATH = ROOT / "Stiles Group Models" / "transformer_v2.keras"

# Initial M2 opponent pool — these originals are NEVER removed.
# Per the assignment, the original copy of M1 is also a valid opponent.
M2_PATHS = {
    "stiles_transformer_orig": ROOT / "Stiles Group Models" / "transformer_v2.keras",
    "stiles_cnn":              ROOT / "Stiles Group Models" / "cnn_v2.keras",
    "luke_cnn":                ROOT / "Luke Group Models"   / "best_cnn_model.keras",
    "luke_transformer":        ROOT / "Luke Group Models"   / "best_transformer_model.keras",
    "zan_cnn":                 ROOT / "Zan Group Models"    / "final_supervised_256f.keras",
    "zan_transformer":         ROOT / "Zan Group Models"    / "transformer.weights.h5",
}

# ── Output directories ────────────────────────────────────────────────────────
CHECKPOINT_DIR = ROOT / "checkpoints"
LOG_DIR        = ROOT / "logs"

# ── Reproducibility ───────────────────────────────────────────────────────────
SEED = 42

# ── Game settings ─────────────────────────────────────────────────────────────
# Random warm-up moves applied before either model takes over.
# Both players alternate during warm-up; no triplets are recorded.
# Helps M1 encounter mid-game positions quickly. Set to 0 to disable.
RANDOM_INIT_MOVES = 4

# ── Policy gradient hyperparameters ──────────────────────────────────────────
GAMES_PER_GROUP = 20    # games per M2 opponent before one gradient step
BATCH_SIZE      = 32    # triplets per gradient step — MUST stay constant (TF warning)
GAMMA           = 0.99  # discount factor applied to terminal reward
LEARNING_RATE   = 1e-4  # Adam optimizer learning rate
NUM_GROUPS      = 500   # total outer training iterations (groups)

# ── Opponent pool settings ────────────────────────────────────────────────────
POOL_CAP          = 20  # maximum pool size; originals always kept regardless
POOL_ADD_INTERVAL = 50  # add a frozen M1 snapshot every this many groups

# ── Checkpointing ─────────────────────────────────────────────────────────────
CHECKPOINT_INTERVAL = 50  # save M1 weights to disk every this many groups

"""
sac_trainer.py — Soft Actor-Critic for discrete Connect-4.

SAC is the genuine PG + Q-learning fusion: policy gradient on the actor side,
off-policy Q-learning with a target network on the critic side. The three
learned components:

  - Policy network  π(a|s)   — 7-column softmax (the "PG" side)
  - Q-network       Q(s, a)  — 7-column linear (the "DQN" side)
  - Target Q-network         — polyak-updated snapshot of Q (DQN stability)

Training objectives (standard SAC-discrete):

  Q-loss:        MSE( Q(s, a), r + γ (1 - done) · Σ_a' π(a'|s') [Q_target(s', a') − α log π(a'|s')] )
  Policy-loss:   E_s [ Σ_a π(a|s) (α log π(a|s) − Q(s, a)) ]
  α (temperature): fixed

Why SAC beats vanilla PG and vanilla AC for this project:

  - Q(s, a) is estimated per action (not just V(s)), trained via Bellman
    bootstrap, so Q-learning is fully off-policy.
  - Replay buffer → old transitions are reused, dramatically better sample
    efficiency than on-policy AC.
  - Target network + entropy regularisation → very stable across hundreds
    of groups where vanilla AC was collapsing.

This trainer is self-contained: everything it needs beyond the shared
modules (game_engine, model_loader, opponent_pool, eval) is defined here.
Intended to be called from notebooks/sac_training.ipynb.
"""

import json as _json
import random as _random
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import tensorflow as tf
# Prefer the plain text-mode tqdm over tqdm.auto — the widget frontend is
# flaky in some Jupyter/Cursor setups ("Error displaying widget: model not
# found") and the text version works everywhere (terminal, Colab, notebooks).
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable=None, **kwargs):
        return iterable if iterable is not None else range(0)

from . import config as _cfg
from . import game_engine as ge
from . import model_loader as ml
from .model_loader import ModelWrapper
from .opponent_pool import OpponentPool
from . import eval as eval_mod


# ─────────────────────────────────────────────────────────────────────────────
# Hyperparameter container
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SACConfig:
    """All hyperparameters for a SAC training run."""
    # Training schedule
    num_groups:        int   = 200          # outer training iterations
    games_per_group:   int   = 32           # games per group (parallel self-play)
    updates_per_group: int   = 16           # SAC gradient steps per group
    batch_size:        int   = 128          # transitions per gradient step
    min_buffer_size:   int   = 2000         # wait for buffer to fill before training

    # Discount + warmup
    gamma:             float = 0.98
    # Right-skewed warmup: probabilities for n = 4..14 (in order). Sums to 1.
    warmup_probs: tuple = (0.25, 0.20, 0.15, 0.10, 0.08, 0.06, 0.05, 0.04, 0.03, 0.02, 0.02)

    # Opponent behaviour
    tactics_prob:      float = 0.5          # probability M2 uses immediate-win/block

    # Optimizer
    learning_rate:     float = 3e-4
    grad_clip_norm:    float = 1.0

    # SAC-specific
    alpha:             float = 0.1          # entropy temperature (fixed)
    tau:               float = 0.005        # soft target-network update rate (Polyak)

    # N-step Bellman bootstrap: target_q = G_n + γ^n · (1-done) · V(s_{t+n}).
    # n=1 is vanilla 1-step TD; higher values give faster credit assignment
    # through the game at the cost of slightly higher variance. n=3 is the
    # sweet spot for sparse-terminal-reward games like Connect-4.
    n_step:            int   = 3

    # Symmetry augmentation: Connect-4 has a horizontal mirror symmetry
    # (column c <-> 6-c is equivalent play). When True, every transition
    # collected from self-play also contributes its mirrored twin to the
    # replay buffer, doubling effective training data at near-zero cost.
    symmetry_augment:  bool  = True

    # Replay buffer
    buffer_capacity:   int   = 50_000

    # Reward shaping: per-move shaping based on net open-three threats.
    # Set shaping_coef=0 to disable.
    shaping_coef:      float = 0.03

    # Opponent pool
    pool_cap:              int   = 12
    pool_add_interval:     int   = 30
    pool_originals_weight: float = 3.0
    pool_snapshot_weight:  float = 1.0

    # Architecture (only used when building a model from scratch)
    conv_filters:      tuple = (64, 128, 128)
    dense_units:       int   = 256
    dropout_rate:      float = 0.1
    q_hidden:          int   = 256

    # Checkpointing
    checkpoint_dir:      Optional[str] = None
    checkpoint_interval: int           = 20
    resume_if_exists:    bool          = True

    # In-training eval
    eval_interval:  int   = 25
    eval_n_games:   int   = 40


# ─────────────────────────────────────────────────────────────────────────────
# Model architecture: dual-head with a Q-head (not a V-head)
# ─────────────────────────────────────────────────────────────────────────────

def build_sac_model(cfg: SACConfig) -> tf.keras.Model:
    """
    Build a dual-head SAC model from scratch:
      - Shared conv trunk
      - Policy head (7 softmax)
      - Q head (7 linear) — one Q-value per column
    """
    inp = tf.keras.Input(shape=(6, 7, 2), name="board")
    x = inp
    for f in cfg.conv_filters:
        x = tf.keras.layers.Conv2D(f, (3, 3), padding="same", activation="relu")(x)
        x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(cfg.dense_units, activation="relu")(x)
    x = tf.keras.layers.Dropout(cfg.dropout_rate)(x)

    policy = tf.keras.layers.Dense(7, activation="softmax", name="policy")(x)

    q_hidden = tf.keras.layers.Dense(cfg.q_hidden, activation="relu")(x)
    q_values = tf.keras.layers.Dense(7, activation="linear", name="q_values")(q_hidden)

    return tf.keras.Model(inputs=inp, outputs=[policy, q_values], name="sac_model")


def build_sac_from_pretrained(
    pretrained_model: tf.keras.Model,
    q_hidden: int = 256,
) -> tf.keras.Model:
    """
    Build a SAC model that warm-starts its policy from a pretrained Connect-4
    policy network (Zan CNN, Stiles CNN, etc.). The Q head is new and random;
    SAC learns it from scratch via Bellman bootstrap, which is fine because
    Q-learning is off-policy and the replay buffer supplies old transitions.

    Looks for the pretrained model's 7-column softmax output to use as the
    policy head, and hooks a new Q head off the same penultimate tensor.
    """
    outs = list(pretrained_model.outputs) if hasattr(pretrained_model, "outputs") \
           else [pretrained_model.output]

    policy_output = None
    for o in outs:
        try:
            shape = o.shape
            if len(shape) >= 1 and shape[-1] == 7:
                policy_output = o
                break
        except Exception:
            continue
    if policy_output is None:
        shapes = [getattr(o, "shape", "?") for o in outs]
        raise ValueError(
            f"Could not find a 7-column policy output in {pretrained_model.name!r}. "
            f"Available output shapes: {shapes}."
        )

    # Find the layer producing the policy, take its input as penultimate
    policy_layer = None
    for layer in pretrained_model.layers:
        try:
            lo = layer.output
            if lo is policy_output:
                policy_layer = layer
                break
        except Exception:
            continue

    if policy_layer is None or not hasattr(policy_layer, "input"):
        penultimate = pretrained_model.layers[-2].output
    else:
        penultimate = policy_layer.input
    if isinstance(penultimate, (list, tuple)):
        penultimate = penultimate[0]

    # Build the Q head on top of the shared feature tensor. Prefix names "sac_"
    # so they cannot collide with a layer the pretrained model already has.
    q_hidden_t = tf.keras.layers.Dense(
        q_hidden, activation="relu", name="sac_q_hidden",
    )(penultimate)
    q_output = tf.keras.layers.Dense(
        7, activation="linear", name="sac_q_values",
    )(q_hidden_t)

    model = tf.keras.Model(
        inputs=pretrained_model.input,
        outputs=[policy_output, q_output],
        name="sac_from_pretrained",
    )
    print("  Warm-started SAC: policy head reused from pretrained; Q head is fresh "
          "(random init, will be trained via Bellman bootstrap).")
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Reward shaping — intermediate signal for creating/blocking open-3 threats
# ─────────────────────────────────────────────────────────────────────────────

def _count_open_threes(board: np.ndarray, player: int) -> int:
    """Count 4-cell windows with exactly 3 of `player`'s pieces + 1 empty.
    Each is a real threat (opponent must block next turn or lose)."""
    count = 0
    for r in range(ge.ROWS):
        for c in range(ge.COLS - 3):
            w = board[r, c:c+4]
            if np.sum(w == player) == 3 and np.sum(w == 0) == 1:
                count += 1
    for r in range(ge.ROWS - 3):
        for c in range(ge.COLS):
            w = board[r:r+4, c]
            if np.sum(w == player) == 3 and np.sum(w == 0) == 1:
                count += 1
    for r in range(ge.ROWS - 3):
        for c in range(ge.COLS - 3):
            w = np.diagonal(board[r:r+4, c:c+4])
            if np.sum(w == player) == 3 and np.sum(w == 0) == 1:
                count += 1
    for r in range(ge.ROWS - 3):
        for c in range(3, ge.COLS):
            w = np.diagonal(np.fliplr(board[r:r+4, c-3:c+1]))
            if np.sum(w == player) == 3 and np.sum(w == 0) == 1:
                count += 1
    return count


def _shaping_delta(board_before: np.ndarray, board_after: np.ndarray,
                   player: int, coef: float) -> float:
    """Per-move shaping reward: (my new threats − opp new threats) × coef."""
    if coef <= 0:
        return 0.0
    my_before  = _count_open_threes(board_before, player)
    my_after   = _count_open_threes(board_after,  player)
    opp_before = _count_open_threes(board_before, -player)
    opp_after  = _count_open_threes(board_after,  -player)
    return coef * ((my_after - my_before) - (opp_after - opp_before))


# ─────────────────────────────────────────────────────────────────────────────
# MinimaxOpponentWrapper — lets minimax sit in the training opponent pool
# ─────────────────────────────────────────────────────────────────────────────

class _MinimaxOpponentCallable:
    """
    Callable that quacks like a Keras model: takes a (B, 6, 7, 2) Type-B
    batch, runs minimax per-sample, returns a (B, 7) one-hot tensor.
    Slow per call (depth-5 is ~50 ms/sample) but only hit on minimax turns.
    """
    def __init__(self, depth: int):
        # Lazy-import so eval doesn't import sac_trainer and vice-versa
        from .eval import MinimaxAgent as _MM
        self.agent = _MM(depth=depth)
        self.depth = depth

    def __call__(self, xs, training=False):
        xs_np = xs.numpy() if hasattr(xs, "numpy") else np.asarray(xs)
        B = xs_np.shape[0]
        out = np.zeros((B, 7), dtype=np.float32)
        for i in range(B):
            # Type-B reconstruction: ch0 = current player, ch1 = opponent.
            ch0 = xs_np[i, :, :, 0].astype(np.int8)
            ch1 = xs_np[i, :, :, 1].astype(np.int8)
            board = ch0 - ch1
            col = self.agent.select_move(board, player=+1)
            out[i, col] = 1.0
        return tf.constant(out)


class MinimaxOpponentWrapper:
    """
    Wraps a MinimaxAgent so it slots into the SAC training opponent pool as
    if it were a ModelWrapper. The training call site doesn't need to know
    the difference — it just calls `.model(xs)` and interprets the output
    as a policy distribution.
    """
    def __init__(self, depth: int, name: Optional[str] = None):
        self.model    = _MinimaxOpponentCallable(depth=depth)
        self.encoding = "B"
        self.name     = name if name is not None else f"minimax_d{depth}"


# ─────────────────────────────────────────────────────────────────────────────
# Replay buffer
# ─────────────────────────────────────────────────────────────────────────────

class ReplayBuffer:
    """Fixed-capacity ring buffer of (s, a, r, s_next, done) transitions."""

    def __init__(self, capacity: int, state_shape=(6, 7, 2)):
        self.capacity     = capacity
        self.states       = np.zeros((capacity, *state_shape), dtype=np.float32)
        self.actions      = np.zeros(capacity,                  dtype=np.int32)
        self.rewards      = np.zeros(capacity,                  dtype=np.float32)
        self.next_states  = np.zeros((capacity, *state_shape), dtype=np.float32)
        self.dones        = np.zeros(capacity,                  dtype=np.float32)
        self.ptr          = 0
        self.size         = 0

    def push(self, s, a, r, s_next, done):
        i = self.ptr
        self.states[i]      = s
        self.actions[i]     = a
        self.rewards[i]     = r
        self.next_states[i] = s_next
        self.dones[i]       = done
        self.ptr            = (self.ptr + 1) % self.capacity
        self.size           = min(self.size + 1, self.capacity)

    def push_many(self, transitions):
        for t in transitions:
            self.push(*t)

    def sample(self, batch_size: int):
        idx = np.random.randint(0, self.size, size=batch_size)
        return (
            self.states[idx],
            self.actions[idx],
            self.rewards[idx],
            self.next_states[idx],
            self.dones[idx],
        )

    def __len__(self):
        return self.size


# ─────────────────────────────────────────────────────────────────────────────
# Helpers: encoding, warmup, opponent sampling, tensor unwrap
# ─────────────────────────────────────────────────────────────────────────────

def _encode_type_b(board: np.ndarray, player: int) -> np.ndarray:
    """Encode a canonical board to (6, 7, 2) Type-B from `player`'s perspective."""
    if player == +1:
        ch0 = (board == +1).astype(np.float32)
        ch1 = (board == -1).astype(np.float32)
    else:
        ch0 = (board == -1).astype(np.float32)
        ch1 = (board == +1).astype(np.float32)
    return np.stack([ch0, ch1], axis=-1)


def _warmup_count(cfg: SACConfig) -> int:
    """Sample a warmup move count from the right-skewed distribution."""
    return int(np.random.choice(range(4, 15), p=np.array(cfg.warmup_probs)))


def _sample_pool(pool: OpponentPool, cfg: SACConfig) -> ModelWrapper:
    """Weighted pool sampling — bias toward originals, away from snapshots."""
    n_orig    = pool._num_original
    pool_size = len(pool)
    if pool_size == n_orig or cfg.pool_originals_weight == cfg.pool_snapshot_weight:
        return pool._pool[np.random.randint(pool_size)]
    w = np.array(
        [cfg.pool_originals_weight] * n_orig
        + [cfg.pool_snapshot_weight] * (pool_size - n_orig),
        dtype=np.float32,
    )
    w /= w.sum()
    return pool._pool[int(np.random.choice(pool_size, p=w))]


def _add_snapshot(pool: OpponentPool, sac_model: tf.keras.Model, cap: int):
    """Add a frozen copy of the current SAC model to the pool (snapshot self-play)."""
    copy_model = tf.keras.models.clone_model(sac_model)
    copy_model.set_weights(sac_model.get_weights())
    snap = ModelWrapper(
        model=copy_model, encoding="B",
        name=f"SAC_snap_{_random.randint(0, 1_000_000):06d}",
    )
    if len(pool) >= cap:
        # Drop the oldest snapshot (originals are always at the front).
        pool._pool = pool._pool[: pool._num_original] + pool._pool[pool._num_original + 1:]
    pool._pool.append(snap)


def _masked_softmax_sample(raw: np.ndarray, legal: list) -> int:
    """Sample a column from raw scores restricted to `legal`."""
    m = np.zeros(7, dtype=np.float32)
    m[legal] = raw[legal]
    s = m.sum()
    if s > 1e-8:
        m /= s
    else:
        m[legal] = 1.0 / len(legal)
    return int(np.random.choice(7, p=m))


def _extract_policy_tensor(out):
    """
    Unwrap arbitrarily nested list/tuple output from model(xs) and return the
    policy tensor (last dim == 7). Falls back to the first leaf if no 7-dim
    output can be identified.
    """
    while isinstance(out, (list, tuple)):
        picked = None
        for x in out:
            s = getattr(x, "shape", None)
            if s is not None and len(s) >= 1 and s[-1] == 7:
                picked = x
                break
        out = picked if picked is not None else out[0]
    return out


def _actor_forward(model: tf.keras.Model, xs: np.ndarray) -> np.ndarray:
    """Extract the (B, 7) policy output from a SAC model's forward pass."""
    pol = _extract_policy_tensor(model(xs, training=False))
    return pol.numpy() if hasattr(pol, "numpy") else np.array(pol)


def _opponent_forward(wrapper: ModelWrapper, xs: np.ndarray) -> np.ndarray:
    """Extract the (B, 7) policy output from an opponent ModelWrapper."""
    pol = _extract_policy_tensor(wrapper.model(xs, training=False))
    return pol.numpy() if hasattr(pol, "numpy") else np.array(pol)


# ─────────────────────────────────────────────────────────────────────────────
# Parallel self-play → SARS(D) transitions for the replay buffer
# ─────────────────────────────────────────────────────────────────────────────

def play_games_and_collect_transitions(
    sac_model: tf.keras.Model,
    m2_wrapper: ModelWrapper,
    cfg: SACConfig,
    use_tactics: bool,
) -> tuple:
    """
    Play cfg.games_per_group games in parallel. For every M1 move, record a
    (state, action, reward, next_state, done) transition:

      - state       : board encoded from M1's perspective BEFORE the move
      - action      : column M1 chose
      - reward      : shaping for this move + terminal reward if the game
                      ends on this move or on the opponent's response
      - next_state  : board encoded from M1's perspective on M1's NEXT turn
                      (per the assignment — s' skips the opponent's move)
      - done        : 1 if the game ended between this M1 move and M1's
                      next turn

    Returns (transitions_list, stats_dict). If symmetry_augment is on, each
    transition is also added as its horizontal mirror (column c <-> 6-c).
    """
    n_games = cfg.games_per_group
    rows, cols = ge.ROWS, ge.COLS

    boards      = [np.zeros((rows, cols), dtype=np.int8) for _ in range(n_games)]
    m1_players  = [int(np.random.choice([+1, -1])) for _ in range(n_games)]
    next_player = [+1] * n_games
    done_flag   = [False] * n_games
    winner      = [None] * n_games

    # Per-game record of M1's turns. Each element is a list of dicts with
    # state (Type-B encoded), action (column), and shaping reward. After the
    # game ends, we walk through this list to produce n-step transitions
    # with (possibly mirrored) data augmentation.
    m1_turns: list = [[] for _ in range(n_games)]

    stats = {"wins": 0, "draws": 0, "losses": 0}

    # Random warmup per game (no transitions recorded during warmup)
    for i in range(n_games):
        n_warm = _warmup_count(cfg)
        boards[i], np_after = ge.random_moves(boards[i], n_warm, +1)
        if np_after is None:
            done_flag[i] = True
        else:
            next_player[i] = np_after

    while not all(done_flag):
        m1_turn, m2_turn = [], []
        for i in range(n_games):
            if done_flag[i]:
                continue
            (m1_turn if next_player[i] == m1_players[i] else m2_turn).append(i)

        # ── M1 side ──────────────────────────────────────────────────────────
        if m1_turn:
            xs = np.stack([
                _encode_type_b(boards[i], m1_players[i]) for i in m1_turn
            ]).astype(np.float32)
            pol = _actor_forward(sac_model, xs)

            for k, i in enumerate(m1_turn):
                legal = ge.legal_moves(boards[i])
                col   = _masked_softmax_sample(pol[k], legal)

                state_t = xs[k].copy()
                board_before = boards[i].copy()
                boards[i] = ge.make_move(boards[i], col, m1_players[i])
                sh = _shaping_delta(board_before, boards[i], m1_players[i],
                                    cfg.shaping_coef)

                m1_turns[i].append({
                    "state":   state_t,
                    "action":  col,
                    "shaping": sh,
                })

                d, w = ge.is_terminal(boards[i])
                if d:
                    if w == m1_players[i]:   stats["wins"]  += 1
                    elif w == 0:             stats["draws"] += 1
                    else:                    stats["losses"] += 1   # unreachable
                    done_flag[i], winner[i] = True, w
                else:
                    next_player[i] = -m1_players[i]

        # ── M2 side ──────────────────────────────────────────────────────────
        need_infer = []
        for i in m2_turn:
            if done_flag[i]:
                continue
            p = -m1_players[i]
            col = None
            if use_tactics:
                col = ge.winning_move(boards[i], p)
                if col is None:
                    col = ge.blocking_move(boards[i], p)
            if col is not None:
                boards[i] = ge.make_move(boards[i], col, p)
                d, w = ge.is_terminal(boards[i])
                if d:
                    if w == -m1_players[i]:  stats["losses"] += 1
                    elif w == 0:             stats["draws"]  += 1
                    else:                    stats["wins"]   += 1   # unreachable
                    done_flag[i], winner[i] = True, w
                else:
                    next_player[i] = m1_players[i]
            else:
                need_infer.append(i)

        if need_infer:
            xs = np.stack([
                ml.encode_board(m2_wrapper, boards[i], -m1_players[i])
                for i in need_infer
            ]).astype(np.float32)
            raw = _opponent_forward(m2_wrapper, xs)
            for k, i in enumerate(need_infer):
                legal = ge.legal_moves(boards[i])
                col   = _masked_softmax_sample(raw[k], legal)
                p     = -m1_players[i]
                boards[i] = ge.make_move(boards[i], col, p)
                d, w = ge.is_terminal(boards[i])
                if d:
                    if w == -m1_players[i]:  stats["losses"] += 1
                    elif w == 0:             stats["draws"]  += 1
                    else:                    stats["wins"]   += 1   # unreachable
                    done_flag[i], winner[i] = True, w
                else:
                    next_player[i] = m1_players[i]

    # ── Build n-step transitions (+ mirror augmentations) from game records ──
    transitions = []
    gamma       = cfg.gamma
    n           = max(1, cfg.n_step)
    augment     = cfg.symmetry_augment

    for i in range(n_games):
        if winner[i] is None:
            continue   # game ended during warmup, no M1 turns to learn from
        turns = m1_turns[i]
        N = len(turns)
        if N == 0:
            continue

        # Per-turn reward: shaping for every turn + terminal outcome added to
        # the LAST M1 turn. Terminal reward is determined by who won the game
        # as a whole — propagated back to the last M1 turn regardless of
        # whether the game ended on M1's or M2's move.
        if winner[i] == m1_players[i]:
            terminal_r = 1.0
        elif winner[i] == -m1_players[i]:
            terminal_r = -1.0
        else:
            terminal_r = 0.0

        per_turn_r = [t["shaping"] for t in turns]
        per_turn_r[-1] += terminal_r

        # For each M1 turn t, compute the n-step discounted return G_t and
        # the bootstrap endpoint state s_{t+n}. If the terminal hit within
        # the n-step window, the bootstrap is zeroed (done=1).
        for t in range(N):
            G = 0.0
            done_in_window = False
            for k_step in range(n):
                idx = t + k_step
                if idx >= N:
                    done_in_window = True
                    break
                G += (gamma ** k_step) * per_turn_r[idx]
                if idx == N - 1:
                    done_in_window = True
                    break

            state_t   = turns[t]["state"]
            action_t  = turns[t]["action"]
            endpoint_idx = t + n
            if done_in_window or endpoint_idx >= N:
                endpoint_state = state_t          # ignored: done=1 zeros bootstrap
                done_val       = 1.0
            else:
                endpoint_state = turns[endpoint_idx]["state"]
                done_val       = 0.0

            # Original transition
            transitions.append((state_t, action_t, G, endpoint_state, done_val))

            # Mirror augmentation (column c <-> 6-c is an equivalent Connect-4
            # position). Doubles effective training data for free.
            if augment:
                mirror_state    = np.flip(state_t,       axis=1).copy()
                mirror_endpoint = np.flip(endpoint_state, axis=1).copy()
                transitions.append((
                    mirror_state, 6 - action_t, G, mirror_endpoint, done_val,
                ))

    return transitions, stats


# ─────────────────────────────────────────────────────────────────────────────
# SAC update — Q-loss (Bellman) + policy-loss (soft policy iteration)
# ─────────────────────────────────────────────────────────────────────────────

def _sac_update(
    sac_model: tf.keras.Model,
    target_model: tf.keras.Model,
    optimizer: tf.keras.optimizers.Optimizer,
    batch: tuple,
    cfg: SACConfig,
) -> dict:
    """
    One SAC gradient step on a sampled batch from the replay buffer.
    Updates Q head and policy head in two separate tape/apply cycles so
    gradients stay clean: the Q-loss only touches Q/trunk variables; the
    policy-loss only touches policy/trunk variables.

    batch: (states, actions, rewards, next_states, dones) numpy arrays.
    """
    s, a, r, s_next, done = batch
    s_tf        = tf.constant(s,       dtype=tf.float32)
    a_tf        = tf.constant(a,       dtype=tf.int32)
    r_tf        = tf.constant(r,       dtype=tf.float32)
    s_next_tf   = tf.constant(s_next,  dtype=tf.float32)
    done_tf     = tf.constant(done,    dtype=tf.float32)

    # ── Compute the N-step Bellman target (no gradient) ──────────────────
    # Each batch row's reward `r` is the n-step discounted return G_n that
    # already sums rewards for t ... t+n-1 with the per-step γ powers folded
    # in. The bootstrap term is therefore multiplied by γ^n, not γ, and
    # V(s') uses the state n steps ahead of s.
    policy_next, q_target_next = target_model(s_next_tf, training=False)
    log_pi_next = tf.math.log(policy_next + 1e-8)
    v_next      = tf.reduce_sum(
        policy_next * (q_target_next - cfg.alpha * log_pi_next),
        axis=-1,
    )
    gamma_n  = cfg.gamma ** max(1, cfg.n_step)
    target_q = r_tf + gamma_n * (1.0 - done_tf) * v_next
    target_q = tf.stop_gradient(target_q)

    # Helper: drop (grad, var) pairs where grad is None before applying.
    # The Q-loss only touches the Q head + trunk (not the policy head) and
    # vice-versa for the policy loss; Keras warns noisily about missing
    # gradients if we pass full trainable_variables with Nones.
    def _apply(grads, variables):
        pairs = [(g, v) for g, v in zip(grads, variables) if g is not None]
        if not pairs:
            return
        gs, vs = zip(*pairs)
        gs, _ = tf.clip_by_global_norm(gs, cfg.grad_clip_norm)
        optimizer.apply_gradients(zip(gs, vs))

    # ── Q-head update ────────────────────────────────────────────────────
    with tf.GradientTape() as tape:
        policy, q_values = sac_model(s_tf, training=True)
        # Q(s, a) for the action actually taken
        q_taken = tf.gather(q_values, a_tf, batch_dims=1)
        q_loss  = tf.reduce_mean(tf.square(q_taken - target_q))

    # NaN guard — a numerically bad batch can otherwise corrupt every weight
    # for the rest of training. Skip the apply; keep the previous weights.
    if not np.isfinite(float(q_loss)):
        tqdm.write("  (NaN/Inf in Q-loss — skipping update)")
        return {"q_loss": float("nan"), "policy_loss": float("nan"), "entropy": float("nan")}
    q_grads = tape.gradient(q_loss, sac_model.trainable_variables)
    _apply(q_grads, sac_model.trainable_variables)

    # ── Policy-head update (Q held fixed) ────────────────────────────────
    with tf.GradientTape() as tape:
        policy, q_values = sac_model(s_tf, training=True)
        q_values_fixed   = tf.stop_gradient(q_values)    # don't flow through Q
        log_pi           = tf.math.log(policy + 1e-8)
        # E_s [ Σ_a π(a|s) (α log π(a|s) − Q(s, a)) ]
        policy_loss = tf.reduce_mean(
            tf.reduce_sum(policy * (cfg.alpha * log_pi - q_values_fixed), axis=-1)
        )

    if not np.isfinite(float(policy_loss)):
        tqdm.write("  (NaN/Inf in policy loss — skipping update)")
        return {"q_loss": float(q_loss), "policy_loss": float("nan"), "entropy": float("nan")}
    pol_grads = tape.gradient(policy_loss, sac_model.trainable_variables)
    _apply(pol_grads, sac_model.trainable_variables)

    # Policy entropy for logging
    entropy = -tf.reduce_mean(tf.reduce_sum(policy * log_pi, axis=-1))

    return {
        "q_loss":      float(q_loss),
        "policy_loss": float(policy_loss),
        "entropy":     float(entropy),
    }


def _polyak_update(target: tf.keras.Model, source: tf.keras.Model, tau: float):
    """Soft-update target weights: θ_target ← τ θ + (1-τ) θ_target."""
    new = [tau * s + (1.0 - tau) * d
           for s, d in zip(source.get_weights(), target.get_weights())]
    target.set_weights(new)


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint save / load
# ─────────────────────────────────────────────────────────────────────────────

def _save_checkpoint(
    ckpt_dir: Path,
    sac_model: tf.keras.Model,
    target_model: tf.keras.Model,
    log: dict,
    eval_history: list,
    group_done: int,
    cfg: SACConfig,
):
    """Write a full checkpoint into `ckpt_dir`. Atomic-ish: state.json is
    written to a tempfile and renamed, so a crashed save can't half-corrupt
    a previous good checkpoint."""
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    sac_model.save(str(ckpt_dir / "sac_model.keras"))
    target_model.save(str(ckpt_dir / "sac_target.keras"))

    state = {
        "group_done":   group_done,        # number of groups ALREADY completed
        "timestamp":    datetime.now().isoformat(),
        "log":          log,
        "eval_history": eval_history,
        "config":       asdict(cfg),
    }
    tmp = ckpt_dir / "state.json.partial"
    with open(tmp, "w") as f:
        _json.dump(state, f, indent=2)
    tmp.replace(ckpt_dir / "state.json")


def _load_checkpoint(ckpt_dir: Path) -> Optional[dict]:
    """Return dict with {sac_model, target_model, log, eval_history, group_done} or None."""
    state_path = ckpt_dir / "state.json"
    if not state_path.exists():
        return None
    sac_path = ckpt_dir / "sac_model.keras"
    tgt_path = ckpt_dir / "sac_target.keras"
    if not (sac_path.exists() and tgt_path.exists()):
        return None
    with open(state_path) as f:
        state = _json.load(f)
    sac_model    = tf.keras.models.load_model(str(sac_path), compile=False)
    target_model = tf.keras.models.load_model(str(tgt_path), compile=False)
    return {
        "sac_model":    sac_model,
        "target_model": target_model,
        "log":          state["log"],
        "eval_history": state["eval_history"],
        "group_done":   state["group_done"],
    }


# ─────────────────────────────────────────────────────────────────────────────
# SACAgent for evaluation / pool play
# ─────────────────────────────────────────────────────────────────────────────

class SACAgent(eval_mod.ModelAgent):
    """
    Plays the SAC policy head greedily (argmax of π(a|s) over legal moves).
    Tactical overrides default ON to mirror tournament deployment. Works with
    the existing eval harness as if it were any ModelAgent.
    """
    def __init__(self, sac_model: tf.keras.Model, name: str = "sac",
                 greedy: bool = True, use_tactics: bool = True):
        wrapper = ModelWrapper(model=sac_model, encoding="B", name=name)
        super().__init__(wrapper, name=name, greedy=greedy, use_tactics=use_tactics)


def _evaluate(sac_model: tf.keras.Model, benchmarks: dict, n_games: int) -> dict:
    """Play the SAC model against each benchmark (greedy, tactics on)."""
    agent = SACAgent(sac_model)
    out = {}
    for name, opponent in benchmarks.items():
        r = eval_mod.play_match_parallel(
            agent, opponent, n_games=n_games, random_init_moves=4, progress=False,
        )
        out[name] = {
            "win_rate":  r.a_win_rate,
            "draw_rate": r.draw_rate,
            "loss_rate": r.b_win_rate,
        }
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Training loop
# ─────────────────────────────────────────────────────────────────────────────

def train(
    cfg: SACConfig,
    initial_pool_wrappers: list,
    benchmarks: Optional[dict] = None,
    initial_model: Optional[tf.keras.Model] = None,
) -> dict:
    """
    Run SAC training. Returns a dict with:
      'model'          : the trained SAC Keras model
      'target_model'   : the target Q network at the end of training
      'log'            : per-group training metrics
      'eval_history'   : list of {group: int, results: dict} from periodic evals
      'elapsed_sec'    : total wall time
      'checkpoint_path': path to the final saved checkpoint dir

    Precedence for resuming:
      - If the caller passes initial_model AND a checkpoint exists, we warn
        loudly and use initial_model (your warm-start wins). Delete the
        checkpoint directory if you want to resume instead.
      - If no initial_model is passed and a checkpoint exists, resume.
      - Otherwise, build a fresh model (or use initial_model if provided).
    """
    ckpt_dir = Path(cfg.checkpoint_dir) if cfg.checkpoint_dir \
               else (_cfg.ROOT / "checkpoints" / "sac_default")
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    resumed = None
    if cfg.resume_if_exists:
        resumed = _load_checkpoint(ckpt_dir)

    if resumed is not None and initial_model is not None:
        print(f"⚠ A checkpoint at {ckpt_dir} exists, AND an initial_model was\n"
              f"  passed to train(). Using the initial_model and IGNORING the\n"
              f"  checkpoint (your warm-start wins). Delete the checkpoint\n"
              f"  directory if you want to resume instead.")
        resumed = None

    if resumed is not None:
        sac_model    = resumed["sac_model"]
        target_model = resumed["target_model"]
        log          = resumed["log"]
        eval_history = resumed["eval_history"]
        start_group  = resumed["group_done"] + 1
        print(f"✓ Resumed SAC from {ckpt_dir} at group {resumed['group_done']} "
              f"(continuing to group {cfg.num_groups})")
        if start_group > cfg.num_groups:
            print(f"  Training already complete ({resumed['group_done']} >= {cfg.num_groups}). "
                  f"Bump cfg.num_groups to train further, or delete the checkpoint to start fresh.")
    else:
        sac_model = initial_model if initial_model is not None else build_sac_model(cfg)
        target_model = tf.keras.models.clone_model(sac_model)
        target_model.set_weights(sac_model.get_weights())
        log = {
            "group": [], "win_rate": [], "draw_rate": [],
            "q_loss": [], "policy_loss": [], "entropy": [],
            "buffer_size": [], "pool_size": [], "opponent": [], "tactics_on": [],
        }
        eval_history = []
        start_group  = 1
        if initial_model is not None:
            print("Using initial_model from caller (no checkpoint to resume from).")

    optimizer = tf.keras.optimizers.Adam(learning_rate=cfg.learning_rate)
    # Pre-build the optimizer with EVERY trainable variable. Keras 3's
    # apply_gradients refuses to accept variables the optimizer wasn't
    # originally built for; since SAC does a Q-update (only Q/trunk vars
    # get gradients) followed by a policy-update (only policy/trunk vars)
    # we need the optimizer to know about both sets from the start.
    optimizer.build(sac_model.trainable_variables)

    # Opponent pool (not persisted across resumes; rebuilt fresh each run)
    pool = OpponentPool(list(initial_pool_wrappers))

    # Replay buffer (also not persisted; rebuilt fresh, filled during run)
    replay = ReplayBuffer(capacity=cfg.buffer_capacity, state_shape=(6, 7, 2))

    t_start = time.time()
    pbar = tqdm(range(start_group, cfg.num_groups + 1),
                desc="SAC training", unit="grp",
                initial=start_group - 1, total=cfg.num_groups)
    for group in pbar:
        # Opponent + tactics for this group
        m2 = _sample_pool(pool, cfg)
        tactics_on = np.random.random() < cfg.tactics_prob

        # Collect transitions
        new_transitions, stats = play_games_and_collect_transitions(
            sac_model, m2, cfg, tactics_on
        )
        replay.push_many(new_transitions)

        # SAC updates (only once the buffer has enough data)
        losses = {"q_loss": 0.0, "policy_loss": 0.0, "entropy": 0.0}
        n_updates = 0
        if len(replay) >= cfg.min_buffer_size:
            for _ in range(cfg.updates_per_group):
                batch = replay.sample(cfg.batch_size)
                step_losses = _sac_update(sac_model, target_model, optimizer, batch, cfg)
                for k in losses:
                    losses[k] += step_losses[k] if np.isfinite(step_losses[k]) else 0.0
                n_updates += 1
                # Soft target update every step
                _polyak_update(target_model, sac_model, cfg.tau)
            for k in losses:
                losses[k] /= max(1, n_updates)
        else:
            losses = {"q_loss": float("nan"), "policy_loss": float("nan"),
                      "entropy": float("nan")}

        total = stats["wins"] + stats["draws"] + stats["losses"]
        win_rate  = stats["wins"]  / total if total else 0.0
        draw_rate = stats["draws"] / total if total else 0.0

        log["group"].append(group)
        log["win_rate"].append(win_rate)
        log["draw_rate"].append(draw_rate)
        log["q_loss"].append(losses["q_loss"])
        log["policy_loss"].append(losses["policy_loss"])
        log["entropy"].append(losses["entropy"])
        log["buffer_size"].append(len(replay))
        log["pool_size"].append(len(pool))
        log["opponent"].append(m2.name)
        log["tactics_on"].append(bool(tactics_on))

        pbar.set_postfix({
            "win%":  f"{win_rate:.0%}",
            "Q":     f"{losses['q_loss']:+.3f}",
            "π":     f"{losses['policy_loss']:+.3f}",
            "H":     f"{losses['entropy']:.2f}",
            "buf":   len(replay),
            "pool":  len(pool),
            "vs":    m2.name[:14],
            "tac":   "Y" if tactics_on else "N",
        })

        if group % cfg.pool_add_interval == 0:
            _add_snapshot(pool, sac_model, cap=cfg.pool_cap)
            tqdm.write(f"  [group {group}] SAC snapshot added to pool (size {len(pool)})")

        if benchmarks is not None and group % cfg.eval_interval == 0:
            eval_res = _evaluate(sac_model, benchmarks, cfg.eval_n_games)
            eval_history.append({"group": group, "results": eval_res})
            summary = "  ".join(f"{n}:{r['win_rate']:.0%}" for n, r in eval_res.items())
            tqdm.write(f"  [group {group}] eval: {summary}")

        if cfg.checkpoint_interval > 0 and group % cfg.checkpoint_interval == 0:
            _save_checkpoint(ckpt_dir, sac_model, target_model, log,
                             eval_history, group, cfg)
            tqdm.write(f"  [group {group}] checkpoint saved → {ckpt_dir}")

    pbar.close()
    elapsed = time.time() - t_start

    # Final save — always writes regardless of checkpoint_interval so the
    # last state is always recoverable.
    final_group = log["group"][-1] if log["group"] else 0
    _save_checkpoint(ckpt_dir, sac_model, target_model, log, eval_history,
                     final_group, cfg)
    print(f"\nFinal checkpoint → {ckpt_dir} (group {final_group}/{cfg.num_groups})")

    return {
        "model":           sac_model,
        "target_model":    target_model,
        "log":             log,
        "eval_history":    eval_history,
        "elapsed_sec":     elapsed,
        "checkpoint_path": str(ckpt_dir),
    }

"""
pg_trainer.py — Policy Gradient training loop (Questions 2 & 3).

High-level flow per training group:
  1. Sample a random M2 from the opponent pool.
  2. Play GAMES_PER_GROUP games of M1 vs M2, collecting (board, player, col, G_t)
     triplets for every M1 move after the random warm-up.
  3. Randomly sample exactly BATCH_SIZE triplets (fixed size — required by TF).
  4. Take one gradient-descent step on M1 using the policy gradient loss.
  5. Checkpoint M1 and maybe add a frozen snapshot to the pool.

Policy gradient loss:
  L = -mean( G_t * log π(a_t | s_t) )
  Maximising expected discounted return ⟺ minimising L.

# DQN trainer will go here (Q4)
"""
import json
import os

import numpy as np
import tensorflow as tf
import wandb

from . import config as _cfg
from . import game_engine as ge
from . import model_loader as ml
from .model_loader import ModelWrapper
from .opponent_pool import OpponentPool


# ─────────────────────────────────────────────────────────────────────────────
# Move selection
# ─────────────────────────────────────────────────────────────────────────────

def _sample_move(wrapper: ModelWrapper, board: np.ndarray, player: int) -> tuple:
    """
    Get the model's softmax probabilities, mask illegal columns, renormalize,
    and sample stochastically.  Returns (col, log_prob).

    This implements the assignment's requirement to "choose a move at random
    with probabilities determined by the output of M1 or M2" rather than
    always picking the argmax.
    """
    moves = ge.legal_moves(board)

    probs = ml.predict_probs(wrapper, board, player)  # raw (7,) softmax

    # Zero out illegal columns, then renormalize
    masked = np.zeros(7, dtype=np.float32)
    masked[moves] = probs[moves]
    total = masked.sum()
    if total <= 1e-8:
        # Fallback: uniform over legal moves if model assigns ~0 mass
        masked[moves] = 1.0 / len(moves)
    else:
        masked /= total

    col = int(np.random.choice(7, p=masked))
    log_prob = float(np.log(masked[col] + 1e-8))
    return col, log_prob


def _sample_move_m2(wrapper: ModelWrapper, board: np.ndarray, player: int) -> tuple:
    """
    Stronger move selection for M2 (per the assignment's adversarial training note):
      1. If M2 can win immediately, play that column.
      2. If M1 would win on the next move, block it.
      3. Otherwise, fall back to stochastic model sampling over legal moves.

    This makes M2 a harder opponent, producing clearer win/loss signals for M1.
    M1's _sample_move is unchanged — M1 learns only through SGD, not rule shortcuts.
    """
    # Immediate win
    col = ge.winning_move(board, player)
    if col is not None:
        return col, 0.0

    # Block opponent's immediate win
    col = ge.blocking_move(board, player)
    if col is not None:
        return col, 0.0

    # Stochastic model sampling
    return _sample_move(wrapper, board, player)


# ─────────────────────────────────────────────────────────────────────────────
# Single game
# ─────────────────────────────────────────────────────────────────────────────

def play_game(m1_wrapper: ModelWrapper, m2_wrapper: ModelWrapper) -> tuple:
    """
    Play one complete game of M1 vs M2.

    - M1's color (+1 or -1) is chosen randomly each game.
    - The first RANDOM_INIT_MOVES moves are played uniformly at random by
      both players (warm-up); no triplets are recorded for these moves.
    - After warm-up, both models sample stochastically from their outputs.
    - Discounted returns are computed from the terminal outcome only
      (intermediate rewards are all 0).

    Returns:
        triplets  : list of (board, m1_player, col, G_t) — one per M1 move
        winner    : +1 / -1 / 0 (draw) / None (game ended during warm-up)
        m1_player : which color M1 was assigned (+1 or -1)
    """
    board = np.zeros((ge.ROWS, ge.COLS), dtype=np.int8)

    # Randomly assign M1 as red (+1) or yellow (-1) for this game
    m1_player = int(np.random.choice([+1, -1]))
    m2_player = -m1_player

    # ── Random warm-up ────────────────────────────────────────────────────────
    # +1 always moves first on an empty board
    next_player = +1
    if _cfg.RANDOM_INIT_MOVES > 0:
        board, next_player = ge.random_moves(board, _cfg.RANDOM_INIT_MOVES, next_player)
        if next_player is None:
            # Game ended during warm-up — no triplets to collect
            return [], None, m1_player

    # ── Main game loop ────────────────────────────────────────────────────────
    m1_boards = []   # canonical boards recorded at each of M1's turns
    m1_moves  = []   # column indices M1 chose

    done   = False
    winner = None

    while not done:
        if next_player == m1_player:
            # M1's turn: sample stochastically and record the state-action pair
            col, _ = _sample_move(m1_wrapper, board, m1_player)
            m1_boards.append(board.copy())
            m1_moves.append(col)
        else:
            # M2's turn: use stronger move selection (win/block before sampling)
            col, _ = _sample_move_m2(m2_wrapper, board, m2_player)

        board = ge.make_move(board, col, next_player)
        done, winner = ge.is_terminal(board)
        next_player = -next_player   # alternate turns

    # ── Compute discounted returns ────────────────────────────────────────────
    # Terminal reward from M1's perspective
    if winner == m1_player:
        r = 1.0
    elif winner == -m1_player:
        r = -1.0
    else:
        r = 0.0   # draw

    # G_t = r * γ^(N−1−t), so moves closer to the end get higher |G_t|
    N = len(m1_moves)
    triplets = []
    for t in range(N):
        G_t = r * (_cfg.GAMMA ** (N - 1 - t))
        triplets.append((m1_boards[t], m1_player, m1_moves[t], G_t))

    return triplets, winner, m1_player


# ─────────────────────────────────────────────────────────────────────────────
# Gradient step
# ─────────────────────────────────────────────────────────────────────────────

def gradient_step(m1_wrapper: ModelWrapper, optimizer, triplets: list) -> float:
    """
    Sample exactly BATCH_SIZE triplets (with replacement to guarantee a fixed
    batch size across all steps) and take one policy-gradient step on M1.

    Loss  =  −mean( G_t * log π(a_t | s_t) )

    Returns the scalar loss value for logging.
    """
    # Sample a fixed-size batch; with replacement so size is always BATCH_SIZE
    indices = np.random.choice(len(triplets), size=_cfg.BATCH_SIZE, replace=True)
    batch   = [triplets[i] for i in indices]

    # Encode each board into M1's expected input format (e.g. (6,7,2) for Type B)
    boards_np = np.stack([
        ml.encode_board(m1_wrapper, board, player)
        for board, player, col, G_t in batch
    ])  # shape: (BATCH_SIZE, 6, 7, 2) for the Stiles Transformer

    actions_np = np.array([col for _, _, col, _   in batch], dtype=np.int32)
    returns_np = np.array([G_t for _, _, _,   G_t in batch], dtype=np.float32)

    # Normalize returns within the batch to reduce gradient variance
    std = returns_np.std()
    if std > 1e-8:
        returns_np = (returns_np - returns_np.mean()) / (std + 1e-8)

    boards_tf  = tf.constant(boards_np,  dtype=tf.float32)
    actions_tf = tf.constant(actions_np, dtype=tf.int32)
    returns_tf = tf.constant(returns_np, dtype=tf.float32)

    with tf.GradientTape() as tape:
        # M1 (Stiles Transformer) has a single output of shape (BATCH_SIZE, 7)
        probs = m1_wrapper.model(boards_tf, training=True)

        # Probability assigned to the action actually taken in each board position
        action_probs = tf.gather(probs, actions_tf, batch_dims=1)  # (BATCH_SIZE,)

        # Policy gradient loss: maximise E[G * log π] ⟺ minimise the negative
        pg_loss = -tf.reduce_mean(returns_tf * tf.math.log(action_probs + 1e-8))

        # Entropy bonus: -H(π) added to loss discourages the policy from collapsing
        # to a small set of moves when it is mostly losing
        entropy = -tf.reduce_mean(
            tf.reduce_sum(probs * tf.math.log(probs + 1e-8), axis=-1)
        )
        loss = pg_loss - _cfg.ENTROPY_COEF * entropy

    grads = tape.gradient(loss, m1_wrapper.model.trainable_variables)

    # Clip gradient norm to guard against reward-signal spikes destabilising training
    grads, _ = tf.clip_by_global_norm(grads, _cfg.GRAD_CLIP_NORM)

    optimizer.apply_gradients(zip(grads, m1_wrapper.model.trainable_variables))

    return float(loss)


# ─────────────────────────────────────────────────────────────────────────────
# Outer training loop
# ─────────────────────────────────────────────────────────────────────────────

def train(m1_wrapper: ModelWrapper, pool: OpponentPool) -> dict:
    """
    Run the full PG training loop for NUM_GROUPS groups.

    Each group:
      1. Sample M2 from the pool.
      2. Play GAMES_PER_GROUP games, taking one gradient step every
         GRAD_STEP_EVERY_N_GAMES games on that fresh mini-batch of triplets.
         No data is reused across gradient steps.
      3. Print every group with a LOG_WINDOW-group rolling average for live monitoring.
      4. Checkpoint M1 every CHECKPOINT_INTERVAL groups, keeping only the
         MAX_CHECKPOINTS most recent files (oldest deleted automatically).
      5. Maybe add a frozen M1 snapshot to the pool every POOL_ADD_INTERVAL groups.

    Returns a log dict with per-group metrics (also saved to logs/).
    """
    os.makedirs(_cfg.CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(_cfg.LOG_DIR,        exist_ok=True)

    wandb.init(
        project="connect4-pg",
        config={
            "learning_rate":           _cfg.LEARNING_RATE,
            "games_per_group":         _cfg.GAMES_PER_GROUP,
            "batch_size":              _cfg.BATCH_SIZE,
            "gamma":                   _cfg.GAMMA,
            "num_groups":              _cfg.NUM_GROUPS,
            "random_init_moves":       _cfg.RANDOM_INIT_MOVES,
            "entropy_coef":            _cfg.ENTROPY_COEF,
            "grad_clip_norm":          _cfg.GRAD_CLIP_NORM,
            "grad_step_every_n_games": _cfg.GRAD_STEP_EVERY_N_GAMES,
            "pool_cap":                _cfg.POOL_CAP,
            "pool_add_interval":       _cfg.POOL_ADD_INTERVAL,
        },
    )

    optimizer = tf.keras.optimizers.Adam(learning_rate=_cfg.LEARNING_RATE)

    log = {
        "group":     [],
        "loss":      [],
        "win_rate":  [],
        "draw_rate": [],
        "pool_size": [],
    }

    checkpoint_files = []  # in-memory list of saved paths, oldest first, for rotation

    for group in range(1, _cfg.NUM_GROUPS + 1):

        # 1. Pick a random M2 opponent for this group
        m2 = pool.sample()

        # 2. Play GAMES_PER_GROUP games; gradient step every GRAD_STEP_EVERY_N_GAMES games
        batch_triplets = []
        step_losses    = []
        wins = draws = losses = 0

        for game_num in range(1, _cfg.GAMES_PER_GROUP + 1):
            triplets, winner, m1_player = play_game(m1_wrapper, m2)
            batch_triplets.extend(triplets)

            if winner == m1_player:
                wins += 1
            elif winner == 0:
                draws += 1
            elif winner is not None:
                losses += 1

            # Take one gradient step on this fresh mini-batch, then discard it
            if game_num % _cfg.GRAD_STEP_EVERY_N_GAMES == 0 and batch_triplets:
                step_losses.append(gradient_step(m1_wrapper, optimizer, batch_triplets))
                batch_triplets = []   # discard — no data reuse

        avg_loss  = sum(step_losses) / len(step_losses) if step_losses else float("nan")
        total     = wins + draws + losses
        win_rate  = wins  / total if total else 0.0
        draw_rate = draws / total if total else 0.0

        log["group"].append(group)
        log["loss"].append(avg_loss)
        log["win_rate"].append(win_rate)
        log["draw_rate"].append(draw_rate)
        log["pool_size"].append(len(pool))

        # 3. Rolling averages over the last LOG_WINDOW groups (x==x filters NaN)
        recent_loss = [x for x in log["loss"][-_cfg.LOG_WINDOW:] if x == x]
        recent_win  = log["win_rate"][-_cfg.LOG_WINDOW:]
        roll_loss   = sum(recent_loss) / len(recent_loss) if recent_loss else float("nan")
        roll_win    = sum(recent_win)  / len(recent_win)  if recent_win  else 0.0

        # Print every group for live monitoring
        print(
            f"[Group {group:4d}] loss={avg_loss:+.3f} | win%={win_rate:.0%} | "
            f"roll(loss)={roll_loss:+.3f} | roll(win%)={roll_win:.0%} | "
            f"vs={m2.name}"
        )

        # Log to Weights & Biases
        wandb.log({
            "loss":       avg_loss,
            "win_rate":   win_rate,
            "draw_rate":  draw_rate,
            "roll_loss":  roll_loss,
            "roll_win_rate": roll_win,
            "pool_size":  len(pool),
        }, step=group)

        # 4. Checkpoint with rotation — keep MAX_CHECKPOINTS most recent files
        if group % _cfg.CHECKPOINT_INTERVAL == 0:
            if len(checkpoint_files) >= _cfg.MAX_CHECKPOINTS:
                oldest = checkpoint_files.pop(0)
                if os.path.exists(oldest):
                    os.remove(oldest)
                    print(f"           → Removed old checkpoint: {os.path.basename(oldest)}")
            ckpt = str(_cfg.CHECKPOINT_DIR / f"m1_group_{group:04d}.keras")
            m1_wrapper.model.save(ckpt)
            checkpoint_files.append(ckpt)
            print(f"           → Saved: {os.path.basename(ckpt)}  ({len(checkpoint_files)}/{_cfg.MAX_CHECKPOINTS} kept)")

        # 5. Maybe add a frozen M1 snapshot to the opponent pool
        added = pool.maybe_add_m1_copy(m1_wrapper, group)
        if added:
            print(f"           → M1 snapshot added to pool (pool size now: {len(pool)})")

    # Persist the training log
    log_path = str(_cfg.LOG_DIR / "training_log.json")
    with open(log_path, "w") as f:
        json.dump(log, f, indent=2)
    print(f"\nTraining complete. Log saved to {log_path}")

    return log

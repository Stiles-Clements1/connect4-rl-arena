# Project 3 — Connect-4 Reinforcement Learning

## Report (Q7)

The final written report is authored in Overleaf:
**[Overleaf project](https://www.overleaf.com/1473459198wdzspxtngrsf#bdb028)**

The LaTeX source skeleton lives at [`report/report.tex`](report/report.tex); figures go in `report/figures/`. Iterate on Overleaf, then sync back to this folder at the end if we want the final `.tex` checked into git.

## What is implemented

This repository covers **Questions 1–3** of the assignment (policy gradient self-play).  
Questions 4–7 (DQN, actor-critic, tournament, report) are for teammates to add.

| Question | Description | Status |
|---|---|---|
| Q1 | Select a pretrained network (M1) | ✓ Stiles Transformer |
| Q2 | Policy gradient training loop | ✓ `pg_trainer.py` |
| Q3 | Opponent pool with M1 snapshots | ✓ `opponent_pool.py` |
| Q4 | DQN training | — see note below |

---

## Folder structure

```
Opti Proj 3/
├── CLAUDE.md                         project-specific AI instructions
├── README.md                         this file
│
├── Stiles Group Models/              pretrained .keras models + backend
├── Luke Group Models/                pretrained .keras models + backend
├── Zan Group Models/                 pretrained .h5/.keras models + backend
│
├── src/                              importable Python modules
│   ├── config.py                     ALL hyperparameters and file paths — edit here
│   ├── game_engine.py                Connect-4 rules (make_move, legal_moves, etc.)
│   ├── model_loader.py               loads all models, handles encoding conversion
│   ├── opponent_pool.py              manages the M2 opponent pool
│   └── pg_trainer.py                 policy gradient loop, gradient step, training
│
├── notebooks/
│   └── project3_pg_training.ipynb   deliverable notebook — thin, imports from src/
│
├── checkpoints/                      M1 snapshots saved during training
└── logs/                             training_log.json + training_curves.png
```

---

## Canonical board representation

All game logic, the training loop, and the opponent pool use a single internal format:

- **numpy array, shape (6, 7), dtype int8**
- `+1` = red player (goes first on an empty board)
- `-1` = yellow player
- `0`  = empty cell
- **Row 0 is the TOP of the board; row 5 is the BOTTOM.**  
  A dropped piece falls to the highest-numbered empty row in its column.  
  (This matches all three groups' backend code — verified by inspection.)

---

## Model encodings

Each group's models were trained with one of two input encodings.  
These are determined by reading the backend source in each group's folder.

| Group | Model file | Encoding type | Input shape |
|---|---|---|---|
| Stiles | `transformer_v2.keras` (M1) | **B** | `(1, 6, 7, 2)` |
| Stiles | `cnn_v2.keras` | **B** | `(1, 6, 7, 2)` |
| Luke | `best_cnn_model.keras` | **A** | `(1, 6, 7, 1)` |
| Luke | `best_transformer_model.keras` | **A** | `(1, 6, 7, 1)` |
| Zan | `final_supervised_256f.keras` | **B** | `(1, 6, 7, 2)` |
| Zan | `transformer.weights.h5` | **B_flat** | `(1, 42, 2)` |

**Type A** — single signed channel. `+1` = current player's piece, `-1` = opponent.  
When playing as `-1`, the board is negated before encoding.

**Type B** — two-channel one-hot. Channel 0 = current player's pieces, channel 1 = opponent's.  
When playing as `-1`, the channels are swapped (perspective flip).

**Type B_flat** — same as Type B but the 6×7 board is flattened to 42 tokens before stacking channels. Used only by Zan's Transformer (weights-only file; architecture rebuilt from `model_wrappers.py`).

---

## How `model_loader` works

`src/model_loader.py` is the **only** module that knows about file extensions or encoding types. Everything else calls:

```python
from src import model_loader

# Load all models at startup
models = model_loader.load_all_models()
# Returns dict: {"m1": ModelWrapper, "stiles_cnn": ModelWrapper, ...}

# Get move probabilities for any model (7-element softmax array)
probs = model_loader.predict_probs(wrapper, board, player)
# board: canonical 6×7 numpy array
# player: +1 or -1

# Get the encoded numpy array for direct model calls (used by gradient tape)
x = model_loader.encode_board(wrapper, board, player)
```

`ModelWrapper` is a small dataclass: `model` (Keras), `encoding` ("A"/"B"/"B_flat"), `name` (string).

Special loading notes handled internally:
- Luke Transformer needs `AddPositionEmb` and `ClassToken` as custom objects (imported from `Luke Group Models/inference.py`).
- Zan Transformer is weights-only: `build_connect4_transformer()` is called first (from `Zan Group Models/model_wrappers.py`), then `load_weights()`.
- The original `transformer_v2.keras` is loaded **twice**: once as M1 (for training) and once as a frozen M2 opponent.

---

## How the opponent pool works

```python
from src import opponent_pool
pool = opponent_pool.OpponentPool(initial_wrappers)
```

- **Seeded** with all non-M1 models plus the original frozen M1 copy (6 models total).
- `pool.sample()` returns a uniformly random opponent for the next group.
- `pool.maybe_add_m1_copy(m1_wrapper, group)` — every `POOL_ADD_INTERVAL` groups, clones M1's current weights into a new frozen wrapper and appends it, provided `len(pool) < POOL_CAP`.
- **Original models are never removed**, regardless of `POOL_CAP`.

---

## How the policy gradient training loop works

```
for group in 1 … NUM_GROUPS:
    M2 = pool.sample()
    for _ in GAMES_PER_GROUP games:
        Randomly assign M1 as +1 or -1
        Apply RANDOM_INIT_MOVES random warm-up moves (no triplets recorded)
        While game not over:
            Current player samples stochastically from its model's softmax
            Record (board, player, col) for M1's turns
        Compute G_t = r · γ^(N−1−t) for each of M1's moves
    Sample BATCH_SIZE triplets (fixed! with replacement)
    Normalize returns within the batch
    One GradientTape step:  loss = −mean(G_t · log π(a_t|s_t))
    Checkpoint M1 every CHECKPOINT_INTERVAL groups
    Maybe add frozen M1 copy to pool every POOL_ADD_INTERVAL groups
```

Moves are chosen **stochastically** (sampled from the softmax distribution over legal moves) — not argmax — so the agent explores. The batch size is held constant across all gradient steps to prevent TensorFlow graph retracing.

---

## Key hyperparameters

All hyperparameters live in `src/config.py`. The most important ones:

| Name | Default | Description |
|---|---|---|
| `GAMES_PER_GROUP` | 20 | Games per M2 opponent per gradient step |
| `BATCH_SIZE` | 32 | Triplets per gradient step — do not vary |
| `GAMMA` | 0.99 | Discount factor for terminal reward |
| `LEARNING_RATE` | 1e-4 | Adam learning rate |
| `NUM_GROUPS` | 500 | Total training iterations |
| `RANDOM_INIT_MOVES` | 4 | Warm-up moves before models take over |
| `POOL_CAP` | 20 | Maximum opponent pool size |
| `POOL_ADD_INTERVAL` | 50 | Groups between M1 snapshots added to pool |
| `CHECKPOINT_INTERVAL` | 50 | Groups between M1 checkpoint saves |

---

## How to run the notebook

1. Open `notebooks/project3_pg_training.ipynb` in Jupyter or VS Code.
2. Run all cells top to bottom.
3. Cell 3 loads all models (~30–60 s on first run).
4. Cell 5 runs training (progress printed every 10 groups).
5. Cell 6 plots the loss curve and win-rate curve and saves them to `logs/`.

Checkpoints are saved to `checkpoints/m1_group_XXXX.keras` automatically.

---

## Where to plug in the DQN (Q4)

The seam is in `src/pg_trainer.py` — see the comment `# DQN trainer will go here (Q4)` at the top of `train()`. The modules you can reuse without modification:

- **`game_engine.py`** — `make_move`, `legal_moves`, `is_terminal`, `random_moves` work for any trainer.
- **`model_loader.py`** — `load_all_models()` and `predict_probs()` are encoding-agnostic; your DQN can call them the same way.
- **`opponent_pool.py`** — `OpponentPool` works for any agent type; just construct it with the wrappers you want.
- **`config.py`** — add DQN-specific hyperparameters (epsilon schedule, replay buffer size, target network update interval) here.

The DQN model itself would likely be a new Keras model with `tanh` output (Q-values) instead of `softmax`. Load it into a `ModelWrapper` with encoding `"B"` and use `encode_board()` to convert boards before feeding it. The state variable is the board on M1's next turn (not the opponent's), consistent with the coin-game framing from class.

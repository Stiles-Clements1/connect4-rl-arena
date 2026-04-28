# Project 3 — Connect-4 Reinforcement Learning

## What is implemented

This repository covers **Questions 1–3** of the assignment (policy gradient self-play).  
Questions 4–6 (DQN, actor-critic, tournament) are still in progress.

| Question | Description | Status |
|---|---|---|
| Q1 | Select a pretrained network (M1) | ✓ Stiles Transformer |
| Q2 | Policy gradient training loop | ✓ `pg_trainer.py` |
| Q3 | Opponent pool with M1 snapshots | ✓ `opponent_pool.py` |
| Q4 | DQN training | ✓ `colab_dqn_v2.ipynb` (Dueling Double DQN + self-play) |
| Q5 | Head-to-head / round-robin model comparison | ✓ `eval.py` (incl. `MinimaxAgent`) + [`evaluation.ipynb`](https://colab.research.google.com/github/Stiles-Clements1/connect4-rl-arena/blob/main/notebooks/evaluation.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Stiles-Clements1/connect4-rl-arena/blob/main/notebooks/evaluation.ipynb) |
| Extension | MCTS-at-inference experiments — does AlphaZero-style tree search on top of SAC improve play? | ✓ [`src/mcts.py`](src/mcts.py) + [`notebooks/mcts_experiments.ipynb`](notebooks/mcts_experiments.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Stiles-Clements1/connect4-rl-arena/blob/main/notebooks/mcts_experiments.ipynb) |
| Tournament | Live-match clickable UI — MCTS+SAC agent, click columns to record opponent moves | ✓ [`notebooks/tournament_play.ipynb`](notebooks/tournament_play.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Stiles-Clements1/connect4-rl-arena/blob/main/notebooks/tournament_play.ipynb) |
| Q7 | Written report | Skeleton at `report/` → [Overleaf](https://www.overleaf.com/1473459198wdzspxtngrsf#bdb028) |

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
│   ├── pg_trainer.py                 policy gradient loop, gradient step, training
│   └── eval.py                       head-to-head + round-robin evaluation (Q5)
│
├── notebooks/
│   ├── project3_pg_training.ipynb   deliverable notebook — thin, imports from src/
│   └── evaluation.ipynb             interactive toggle UI for head-to-head / round-robin
│
├── report/                           LaTeX source for the Q7 written report
│   ├── report.tex                    skeleton; kept in sync with Overleaf
│   └── figures/                      .png / .pdf figures referenced from report.tex
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

## How evaluation works (Q5)

`src/eval.py` provides a library for comparing any two (or more) agents head-to-head, and `notebooks/evaluation.ipynb` is the interactive UI that drives it.

```python
from src.eval import ModelAgent, RandomAgent, play_match, run_round_robin

# Wrap a ModelWrapper as an Agent. Defaults below mirror tournament deployment.
a = ModelAgent(m1_wrapper,          greedy=True,  use_tactics=True)
b = ModelAgent(luke_transformer_w,  greedy=True,  use_tactics=True)

result = play_match(a, b, n_games=100)   # alternates first player, reports W/L/D
```

Each `ModelAgent` selects moves as follows: (1) if a one-move win is available, play it; (2) if the opponent has a one-move winning threat, block it; (3) otherwise consult the model. The two tactical overrides only play legal moves — they reflect what any competent Connect-4 player does, and match how the agent will behave at tournament time. Turn them off with `use_tactics=False` if you want a model-only comparison for analysis.

The notebook opens with one checkbox per agent, an **N-games** slider, and a **random-init-moves** slider. Selecting exactly **two** boxes runs a head-to-head match; selecting **three or more** runs a round-robin and prints both a win-rate matrix and an overall ranking by mean win rate across opponents.

### Why you should set random-init-moves > 0

Every `ModelAgent` and `MinimaxAgent` in this repo is **deterministic** by default (greedy + tactical overrides). If every game starts from the empty board, a 100-game match between two deterministic agents reduces to only two distinct games (one with each side moving first), and win rates snap to 0% / 50% / 100%. Setting the slider to 4-6 random warm-up moves per game diversifies the starting positions so every game is an independent data point and win rates become continuous (e.g. 58%, 47%, 63%). For report numbers, run with `random_init_moves >= 4`.

### Calibrated minimax baselines

`eval.py` also ships a `MinimaxAgent(depth=N)` — alpha-beta search over a 4-window heuristic with centre-out move ordering. Three depths are pre-built in the notebook and appear as their own checkboxes:

| Agent | Strength |
|---|---|
| `minimax_d1` | Barely tactical (one-move lookahead only). A notch above random. |
| `minimax_d3` | Sees 2–3 ply threats. Beats random easily; beats weak networks. |
| `minimax_d5` | Strong. Beats most humans. Good calibration anchor for the report. |

These give an absolute strength yardstick — "PG beats depth-3 minimax 60% of the time" is more meaningful than "PG beats random 95% of the time." They are also deterministic, so results are reproducible.

### Persisting evaluation results

After every head-to-head or round-robin, the notebook auto-saves two artifacts so nothing is lost when a Colab session ends:

- **`logs/eval_<timestamp>_<tag>.json`** — raw match results (all counts, win rates, draw rates, first-player breakdowns, plus hardware + settings metadata). Re-load later with plain `json.load`.
- **`report/figures/win_rate_matrix_<timestamp>_<tag>.png`** — heatmap of the round-robin, sized and annotated for dropping straight into the Q7 report.

You can also call the helpers programmatically:

```python
from src.eval import save_results_json, save_win_rate_heatmap

json_path = save_results_json(results, tag="pg_vs_baselines", metadata={"n_games": 100})
png_path  = save_win_rate_heatmap(df, title="PG vs minimax")
```

Under Colab, remember to download these files before the runtime disconnects — they live inside `/content/connect4-rl-arena/` on the VM and disappear with the runtime otherwise. For the report, commit the PNG under `report/figures/` and reference it in `report.tex`.

### Where the large Zan CNN comes from

`final_supervised_256f.keras` is 226 MB — too large for git. `src/model_loader.py` resolves it in three places, in order:

1. `Zan Group Models/final_supervised_256f.keras` in the repo (if you put it there manually).
2. A per-user cache at `~/.keras/connect4_rl_arena/final_supervised_256f.keras`.
3. The [GitHub Release `models-v1`](https://github.com/Stiles-Clements1/connect4-rl-arena/releases/tag/models-v1) — downloaded to the cache once, reused thereafter.

So first time someone clones the repo (or opens the Colab notebook on a fresh runtime), loading the Zan CNN triggers a one-time ~30-second download. Every run after that uses the cached copy. If all three locations fail, the model is skipped with a clear message and the other five pretrained networks still load.

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

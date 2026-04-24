# Project 3 — Connect-4 Reinforcement Learning

Self-play reinforcement learning for Connect-4, covering the full set of
assignment questions. Team: Stiles Clements, Luke Hartfield, Alina Hota,
Zan Merrill.

## What is implemented

| Question | Description | Status | Where |
|---|---|---|---|
| Q1 | Select a pretrained network (M1) | ✓ | Stiles Transformer (`Stiles Group Models/transformer_v2.keras`) |
| Q2 | Policy gradient training loop | ✓ | [`src/pg_trainer.py`](src/pg_trainer.py) + [`notebooks/project3_pg_training.ipynb`](notebooks/project3_pg_training.ipynb) [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Stiles-Clements1/connect4-rl-arena/blob/main/notebooks/project3_pg_training.ipynb) |
| Q3 | Opponent pool with M1 snapshots | ✓ | [`src/opponent_pool.py`](src/opponent_pool.py) |
| Q3+ | SAC (PG + Q-learning fused) — final submission | ✓ | [`src/sac_trainer.py`](src/sac_trainer.py) + [`notebooks/sac_training.ipynb`](notebooks/sac_training.ipynb) [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Stiles-Clements1/connect4-rl-arena/blob/main/notebooks/sac_training.ipynb) |
| Q4 | DQN training | ✓ | [`notebooks/colab_simple_dqn.ipynb`](notebooks/colab_simple_dqn.ipynb) [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Stiles-Clements1/connect4-rl-arena/blob/main/notebooks/colab_simple_dqn.ipynb) + `RL models/enhanced_dqn_optimized.h5` |
| Q5 | Head-to-head / round-robin comparison | ✓ | [`src/eval.py`](src/eval.py) + [`notebooks/evaluation.ipynb`](notebooks/evaluation.ipynb) [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Stiles-Clements1/connect4-rl-arena/blob/main/notebooks/evaluation.ipynb) |
| Q7 | Written report | In progress | [`report/report.tex`](report/report.tex) → [Overleaf](https://www.overleaf.com/1473459198wdzspxtngrsf#bdb028) |

---

## Folder structure

```
Opti Proj 3/
├── CLAUDE.md                         project-specific AI instructions
├── README.md                         this file
├── .gitignore
│
├── Stiles Group Models/              pretrained .keras models + Project-1 backend code
├── Luke Group Models/                pretrained .keras models + Project-1 backend code
├── Zan Group Models/                 pretrained .keras/.h5 models + Project-1 backend code
│
├── RL models/                        ⭐  FINISHED, tournament-ready trained agents
│   ├── README.md                     what lives here, overwrite-in-place rules
│   ├── sac_zan.keras                 Zan's SAC submission (~80% mean win rate)
│   └── enhanced_dqn_optimized.h5     Alina's DQN submission
│
├── src/                              importable Python modules
│   ├── config.py                     hyperparameters + file paths (edit here)
│   ├── game_engine.py                Connect-4 rules (make_move, legal_moves, ...)
│   ├── model_loader.py               loads all models, handles encoding conversion,
│   │                                 auto-discovers any .keras/.h5 in the repo
│   ├── opponent_pool.py              manages the M2 opponent pool
│   ├── pg_trainer.py                 Q2-Q3 policy gradient loop
│   ├── sac_trainer.py                SAC trainer used for the final Zan submission
│   └── eval.py                       head-to-head + round-robin (Q5) + MinimaxAgent
│
├── notebooks/
│   ├── project3_pg_training.ipynb    Q1-Q3 baseline PG trainer
│   ├── sac_training.ipynb            SAC trainer (final submission notebook)
│   ├── colab_simple_dqn.ipynb        Alina's DQN trainer (Q4)
│   └── evaluation.ipynb              interactive eval UI (Q5)
│
├── checkpoints/                      training-time model snapshots (committed)
│   └── sac_run/                      latest SAC run — re-running notebooks/sac_training.ipynb
│       ├── sac_model.keras           resumes from here automatically.
│       ├── sac_target.keras
│       └── state.json
│
├── logs/
│   ├── sac_training_notes.md         ⭐  full SAC methodology + results write-up
│   ├── sac_training_curves.png       training plots
│   ├── training_log.json             PG training log
│   ├── training_curves.png           PG training plots
│   ├── enhanced_dqn_optimized_log.json  DQN training log
│   └── grid_search_*.{json,png}      PG hyperparameter sweep
│
└── report/
    ├── report.tex                    LaTeX skeleton; kept in sync with Overleaf
    └── figures/                      .png heatmaps referenced from report.tex
```

### ⭐ Where to look first

| What you want | Where |
|---|---|
| Run the final SAC model in a Colab | `notebooks/sac_training.ipynb` (click the Colab badge above) |
| Compare agents head-to-head | `notebooks/evaluation.ipynb` (click the Colab badge) |
| The full SAC methodology / training story | `logs/sac_training_notes.md` |
| Pretrained + trained models | `Stiles/Luke/Zan Group Models/`, `RL models/` |

---

## Canonical board representation

All game logic, training loops, and the opponent pool use one internal format:

- **numpy array, shape (6, 7), dtype int8**
- `+1` = red player (goes first on an empty board)
- `-1` = yellow player
- `0`  = empty cell
- **Row 0 is the TOP of the board; row 5 is the BOTTOM.** A dropped piece
  falls to the highest-numbered empty row in its column. (Matches all
  three groups' backend code — verified by inspection.)

---

## Model encodings

Each group's pretrained models were trained with one of three input encodings,
determined by reading the backend source in each group's folder.

| Group | Model file | Encoding | Input shape |
|---|---|---|---|
| Stiles | `transformer_v2.keras` (M1) | **B** | `(6, 7, 2)` |
| Stiles | `cnn_v2.keras` | **B** | `(6, 7, 2)` |
| Luke | `best_cnn_model.keras` | **A** | `(6, 7, 1)` |
| Luke | `best_transformer_model.keras` | **A** | `(6, 7, 1)` |
| Zan | `final_supervised_256f.keras` | **B** | `(6, 7, 2)` |
| Zan | `transformer.weights.h5` | **B_flat** | `(42, 2)` |
| (SAC / DQN trained models) | `RL models/*.keras / *.h5` | **B** | `(6, 7, 2)` |

- **Type A** — single signed channel. `+1` = current player's piece, `-1` = opponent.
  When playing as `-1`, the board is negated before encoding.
- **Type B** — two-channel one-hot. Channel 0 = current player's pieces,
  channel 1 = opponent's. When playing as `-1`, the channels are swapped.
- **Type B_flat** — same as Type B but the 6×7 board is flattened to 42 tokens
  before stacking channels. Used only by Zan's Transformer.

Encoding conversion happens only inside `model_loader`, at the moment a
model is called. Every other module stays encoding-agnostic.

---

## How `model_loader` works

`src/model_loader.py` is the **only** module that knows about file
extensions or encoding types. Everything else calls:

```python
from src import model_loader

# Load every configured model + auto-discover .keras/.h5 anywhere in the repo
models = model_loader.load_all_models_with_discovery()
# models["m1"] = Stiles transformer, models["zan_cnn"] = Zan CNN, ...
# models["sac_zan"] = the trained SAC model (discovered from RL models/)
# models["enhanced_dqn_optimized"] = Alina's DQN (discovered from RL models/)

# Get move probabilities for any model (7-element array)
probs = model_loader.predict_probs(wrapper, board, player)

# Get the encoded numpy array for direct model calls (used by gradient tape)
x = model_loader.encode_board(wrapper, board, player)
```

**Auto-discovery.** Drop any `.keras` or `.h5` file into `RL models/`,
`checkpoints/`, or any folder under the repo root, and it will show up as
an opponent in the evaluation notebook on the next run. Files whose names
contain `target`, `snapshot`, or `.partial` are skipped (they're training
byproducts, not deployable agents).

**Large files over the GitHub 100 MB git limit** (currently just the
226 MB Zan CNN) are hosted as GitHub Release assets; `model_loader.py`
auto-downloads and caches them on first use.

---

## How the opponent pool works

```python
from src.opponent_pool import OpponentPool
pool = OpponentPool(initial_wrappers)
```

- Seeded with all non-M1 models (originals that are never removed).
- `pool.sample()` returns a uniformly random opponent.
- `pool.maybe_add_m1_copy(m1_wrapper, group)` — every
  `POOL_ADD_INTERVAL` groups, clones M1's current weights into a new
  frozen wrapper and appends it, provided `len(pool) < POOL_CAP`.
- **Originals are never removed**, regardless of `POOL_CAP`.

The SAC trainer extends this with weighted sampling (originals weighted
more heavily than snapshots) — see `SACConfig.pool_originals_weight`.

---

## How training works

### PG (Q1–Q3)

```
for group in 1 … NUM_GROUPS:
    M2 = pool.sample()
    for _ in GAMES_PER_GROUP games:
        Assign M1 as +1 or -1 randomly
        Apply RANDOM_INIT_MOVES random warm-up moves (no triplets recorded)
        Self-play, sampling stochastically from π
        Record (board, player, col) for each M1 move
        Compute G_t = r · γ^(N−1−t)
    Take one gradient step on the batch:  L = −mean(G_t · log π(a_t|s_t))
    Checkpoint M1 every CHECKPOINT_INTERVAL groups
    Maybe add frozen M1 snapshot to pool every POOL_ADD_INTERVAL groups
```

### SAC (final submission)

Policy + Q + target Q, off-policy with a replay buffer and N-step Bellman
bootstrap (n=3). Full methodology in [`logs/sac_training_notes.md`](logs/sac_training_notes.md).
Short version:

```
for group in 1 … num_groups:
    M2 = weighted pool sample (originals × 3, snapshots × 1)
    Collect transitions via parallel self-play
    Augment with horizontal mirrors, push to replay buffer
    For updates_per_group SAC steps:
        Sample batch, compute n-step Bellman target
        Q-loss + policy-loss with entropy bonus
        Polyak soft-update target network (τ = 0.005)
    Every pool_add_interval groups: add frozen SAC snapshot to pool
    Every checkpoint_interval groups: save checkpoint + state.json
```

### DQN (Alina, Q4)

Independent trainer in `notebooks/colab_simple_dqn.ipynb`. Epsilon-greedy
exploration, replay buffer, target network, progressive-difficulty
opponent. Saves to `checkpoints/enhanced_dqn_optimized.h5` during training;
the promoted final model lives at `RL models/enhanced_dqn_optimized.h5`.

---

## How to run

**Every notebook runs on both Google Colab and locally without edits.** Cell 1
of each notebook detects the environment automatically — on Colab it clones
(or pulls) the repo into `/content/connect4-rl-arena`; on local it walks up
from the notebook's CWD to find the repo root.

### On Colab (recommended for training)

Click any of the Colab badges above. The notebook clones `main` on first
run, installs `tqdm` + `ipywidgets`, and Cell 1 does the rest. Enable a
GPU: Runtime → Change runtime type → GPU (T4 / A100).

### Locally (Jupyter, VS Code, Cursor)

From the repo root:

```
jupyter notebook notebooks/evaluation.ipynb
```

or open the `.ipynb` file in VS Code / Cursor and run cells. Working
directory doesn't matter — Cell 1 walks up to find the repo root.

### Hyperparameters

- **PG:** `src/config.py` — edit once, PG notebook uses these values.
- **SAC:** `SACConfig(...)` in Cell 5 of `notebooks/sac_training.ipynb`.
- **DQN:** top of `notebooks/colab_simple_dqn.ipynb`.

### Promoting a trained model

Every training notebook has a **commented** "save finished model to
`RL models/`" cell at the bottom with a ⚠️ warning. Uncomment it and run
once you're happy with a run. Same-name files are overwritten, so
re-training iteratively just updates the same `RL models/*.keras` file.

---

## Canonical file locations (contract with teammates)

| Kind | Path | Behaviour |
|---|---|---|
| Finished / deployable models | `RL models/` | Overwrite-in-place. Always the tournament-ready version. |
| Training-time checkpoints | `checkpoints/` | Overwrite-in-place. Each trainer has its own subfolder (e.g. `checkpoints/sac_run/`). |
| Training logs + plots | `logs/` | Append as new files (timestamped); never overwrite other runs. |
| Report figures | `report/figures/` | Auto-populated by the evaluation notebook. |

---

## Where to plug in the DQN (Q4 seam)

For teammates working on alternative DQN variants:

- **`src/game_engine.py`** — `make_move`, `legal_moves`, `is_terminal`,
  `random_moves` work for any trainer. Reusable as-is.
- **`src/model_loader.py`** — `load_all_models()` and
  `load_all_models_with_discovery()` return encoding-aware `ModelWrapper`
  instances. Any DQN using a `(6, 7, 2)` two-channel input gets
  `encoding="B"` and plugs into the eval harness without code changes.
- **`src/opponent_pool.py`** — `OpponentPool` is trainer-agnostic.
- **`src/config.py`** — add DQN-specific hyperparameters here.
- **`src/eval.py`** — `ModelAgent` treats DQN Q-values exactly like
  softmax probs under `greedy=True` (argmax of the raw 7-column output).
  If your DQN outputs negative Q-values and you want stochastic sampling,
  shift them positive or softmax-over-temperature before sampling.

The cleanest seam: write `src/dqn_trainer.py` next to `sac_trainer.py`,
following the same `train(cfg, initial_pool_wrappers, ...)` interface, and
add a notebook in `notebooks/` that mirrors `sac_training.ipynb`. Alina's
`notebooks/colab_simple_dqn.ipynb` is the inline-everything version; the
module-plus-thin-notebook pattern is optional.

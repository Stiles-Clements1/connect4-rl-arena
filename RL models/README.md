# RL models — finished, deployable agents

This folder holds the **finished, tournament-ready** models our team has
trained. It is the canonical place to point the evaluation harness at when
you want to compare final agents — not works-in-progress.

## What lives here

| File | Who trained it | Method | Encoding | Notes |
|---|---|---|---|---|
| `sac_zan.keras` | Zan | Soft Actor-Critic (Q3) | B | ~80% mean win rate vs. the baseline pool. Notes in [`logs/sac_training_notes.md`](../logs/sac_training_notes.md). |
| `enhanced_dqn_optimized.h5` | Alina | Enhanced DQN (Q4) | B | Training log at [`logs/enhanced_dqn_optimized_log.json`](../logs/enhanced_dqn_optimized_log.json). |

Additional finished models will land here as teammates complete their
training runs (PG final, minimax-distilled, tournament submission, etc.).

## Rules for this folder

1. **Overwrite-in-place is expected.** If you train a new and better
   version of `sac_zan.keras`, drop it on top of the old one. Git will
   track the change; there is no need to version-bump the filename.
2. **Only put finished models here.** Mid-training checkpoints live in
   `../checkpoints/` — those can change at any time and are not guaranteed
   to be loadable by the eval harness.
3. **Don't delete files unless you're cleaning up your own model.**
   Removing a teammate's model will break their evaluation runs.

## How the rest of the repo uses it

`src/model_loader.py` automatically scans this folder (and the rest of
the repo) for `.keras` / `.h5` files and loads each into a `ModelWrapper`.
Any model you drop here is available as an opponent in the evaluation
notebook on the next run — no code edits required.

```python
# From a notebook:
from src import model_loader
models = model_loader.load_all_models_with_discovery()
# Models in "RL models/" appear under their filename stem:
sac_agent = models["sac_zan"]        # ModelWrapper, encoding="B"
dqn_agent = models["enhanced_dqn_optimized"]
```

See the root `README.md` for the full evaluation workflow.

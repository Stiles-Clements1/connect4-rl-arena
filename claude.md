# claude.md — Opti Proj 3 (Connect-4 Reinforcement Learning)

> **Read this file in full before doing anything else. Keep its contents in context for the entire project.**

---

## 1. Project Context

This is **Optimization Project 3**, a Connect-4 self-play reinforcement learning project. The full assignment instructions are in the project root as:

```
project 3 overview - connect 4 new.pdf
```

**Before writing any code or any plan, read that PDF.** It is the source of truth for what the project requires. This `claude.md` file only adds structure and constraints on top of it.

The working directory when opened in Cursor is the project root: **`Opti Proj 3`**. Both `claude.md` and `README.md` live at the root.

---

## 2. Scope — What I Am Actually Building

**I am only implementing questions 1, 2, and 3 of the assignment.** These cover:

1. Selecting an existing Connect-4 neural network as **M1** and preparing to improve it.
2. Implementing a **policy gradient (PG)** self-play training loop where M1 plays randomly-chosen opponents (M2), collects discounted-reward trajectories, and takes gradient steps.
3. Repeating step 2 across many groups of games, maintaining an **opponent pool** that occasionally absorbs updated copies of M1 (with a size cap, and with the original MCTS-mimicking networks never removed).

**I am NOT implementing questions 4, 5, 6, or 7.** That means:
- No DQN, target network, replay buffer, epsilon-greedy code.
- No actor-critic, minimax, or tournament code.
- No report writing.

**However**, my teammates will implement questions 4+ starting from my code. So the code must be **structured so that adding a DQN later is natural**, not a rewrite. Specifically:
- The game engine, board representation, model loader, and opponent pool must be reusable by a future DQN trainer without modification.
- Do not hardcode "policy gradient" assumptions into shared modules.
- Where a DQN would plug in, a brief comment like `# DQN trainer will go here (Q4)` is welcome. **Do not write stub functions or empty classes for future work** — just leave the seam clean.

---

## 3. Workflow Rules (Plan → Confirm → Code)

**Never write code before presenting a plan and getting my explicit confirmation.**

The workflow is:

1. When I give a task, Claude responds with a **numbered plan** describing what it will do, what files it will create or modify, and any new packages it wants to import.
2. I review the plan. I may approve it, or I may request edits.
3. If I edit, Claude produces a **revised numbered plan** reflecting my changes. No code yet.
4. Only after I explicitly say something like "approved" or "go ahead" does Claude write code.
5. After coding, Claude updates `README.md` if the logical design changed (see Section 8).

**If anything is ambiguous — about the instructions PDF, a model's encoding, folder layout, or my intent — Claude asks me before assuming.** Guessing is worse than a one-line clarifying question.

---

## 4. Models and the Model Loader

### 4.1 Available models

Three folders at the project root contain pretrained Connect-4 networks from Project 1:

- `Stiles Group Models/` — contains a CNN and a Transformer, both as **`.keras`** files.
- `Luke Group Models/` — contains a CNN and a Transformer, both as **`.keras`** files.
- `Zan Group Models/` — contains a CNN and a Transformer, both as **`.h5`** files.

Each folder also contains the **backend / training code** that was used to build those models. That backend code is the authoritative source for determining the model's input encoding.

### 4.2 M1 is fixed

**M1 is `Stiles Group Models/transformer_v2.keras`.** This is the model I will improve via policy gradient for the entire project. Do not substitute a different model for M1.

All other models (CNNs and Transformers across all three group folders, including the *original* copy of `transformer_v2.keras`) are candidates for the **M2 opponent pool**.

### 4.3 Input encodings

Each model was trained with one of two board encodings. Which one must be determined **by inspecting that model's backend code in its folder** — not guessed from the filename.

- **Type A (single-channel, signed):** a 6×7 matrix where `+1` = red chip, `-1` = yellow chip, `0` = empty. Shape: `(6, 7)` or `(6, 7, 1)`.
- **Type B (two-channel, one-hot by color):** a 6×7×2 tensor. One channel is the red plane (`1` where red, else `0`), the other is the yellow plane (`1` where yellow, else `0`). Shape: `(6, 7, 2)`.

### 4.4 Required abstraction: `model_loader`

Build a single **`model_loader`** module (or notebook section) that is the *only* place in the codebase that knows about file extensions or encoding types. It must:

1. Load any model regardless of `.keras` vs `.h5`.
2. For each model, determine and record its encoding type (A or B) by inspecting the corresponding backend code in its folder. Store this alongside the loaded model (e.g., in a small dict or lightweight wrapper).
3. Expose a uniform interface — something like `predict(board, model_entry)` or a wrapper object with a `.predict(board)` method — that takes a board in the **canonical internal representation** (see Section 5) and handles the encoding conversion internally before calling the underlying Keras model.

The rest of the codebase must never call `keras.models.load_model` directly and must never branch on encoding type. All of that lives in `model_loader`.

If a model folder's encoding cannot be determined confidently from its backend code, **stop and ask me** rather than guessing.

---

## 5. Canonical Internal Board Representation

Inside the game engine, PG training loop, reward calculation, and opponent pool, boards are always represented in **one single canonical format**. Use **Type A** as canonical:

- 6×7 numpy array, `int8` or `int`.
- `+1` = red, `-1` = yellow, `0` = empty.
- Row 0 is the top of the board, row 5 is the bottom (or whichever convention the existing backends use — match them; document the choice in `README.md`).

Encoding conversion to Type B happens **only inside `model_loader`**, at the moment a model is called. This keeps every other module encoding-agnostic.

---

## 6. Project Structure (Starting Suggestion)

Claude may revise this if a better structure emerges, but must keep things organized into logical folders and **update `README.md`** whenever the structure changes.

```
Opti Proj 3/
├── claude.md
├── README.md
├── project 3 overview - connect 4 new.pdf
│
├── Stiles Group Models/          (provided, read-only in spirit)
├── Luke Group Models/            (provided, read-only in spirit)
├── Zan Group Models/             (provided, read-only in spirit)
│
├── src/                          (reusable Python modules, importable from the notebook)
│   ├── game_engine.py            (Connect-4 rules, legal moves, win detection)
│   ├── model_loader.py           (loads .keras/.h5, handles Type A/B encoding)
│   ├── opponent_pool.py          (manages the M2 pool and sampling)
│   ├── pg_trainer.py             (policy gradient loop, discounted rewards, SGD step)
│   └── config.py                 (hyperparameters, paths, seeds)
│
├── notebooks/
│   └── project3_pg_training.ipynb  (the deliverable notebook — thin, imports from src/)
│
├── checkpoints/                  (M1 snapshots saved during training)
│
└── logs/                         (training curves, win rates, reward history)
```

**The deliverable is a `.ipynb` file.** Keep the notebook thin: imports from `src/`, runs training, shows results with plots and printouts. Heavy logic lives in `src/` modules so the notebook stays readable for the professor.

---

## 7. Code Quality Rules

- **Minimum viable product.** Write the least code that correctly implements questions 1–3. No speculative features, no "just in case" utilities, no over-engineering. The audience is my professor; clarity beats cleverness.
- **Comment generously.** The grading rubric rewards readable code. Every non-trivial function gets a short docstring; tricky lines get inline comments.
- **No silent dependencies.** If Claude wants to use a package beyond numpy/tensorflow/keras/matplotlib, it must list it in the plan and get approval before importing it.
- **Configurable, not hardcoded.** Hyperparameters (learning rate, batch size, discount factor, number of games per group, opponent pool cap, etc.) live in `config.py` or a clearly marked cell at the top of the notebook — not scattered through the code.
- **Reproducibility.** Set and expose random seeds (numpy, tensorflow, python `random`) in one place.
- **Checkpoint M1 frequently** during training so a crash doesn't lose progress and so I have snapshots to analyze later.
- **Legal/winning-move helpers from Project 1** may be used for M2 (to make opponents stronger) but should generally **not** be used to force M1's moves — M1 should learn through SGD. This mirrors the assignment's note on adversarial training.
- **Batch size must stay constant across gradient steps** (the assignment explicitly warns about this TF slowdown).

---

## 8. README.md — Keep It Current

Claude maintains a `README.md` at the project root whose job is to let a teammate picking up Question 4 understand my code in a few minutes. It must stay in sync with the code: whenever the logical design changes, update the README in the same turn.

The README should be simple and cover:

- What the project does and what portion is implemented (questions 1–3).
- The folder structure and what lives where.
- The canonical board representation and the two encoding types.
- How `model_loader` works and how to load a model.
- How the opponent pool works (sampling, adding copies of M1, the size cap, the never-remove rule for originals).
- How the PG training loop works at a high level (game → discounted rewards → sampled triplets → one SGD step).
- Key hyperparameters and where to change them.
- How to run the notebook end-to-end.
- A short "where to plug in the DQN" note for teammates working on Question 4.

The README is for humans — my teammates — so prose and short examples beat exhaustive API dumps.

---

## 9. Things Claude Should Ask Me About (Not Guess)

- Any ambiguity in the PDF instructions.
- A model whose encoding type can't be confidently determined from its backend code.
- Any desire to add a new dependency.
- Any proposed change to the folder structure.
- Any place where the "minimum viable" interpretation is unclear.

---

## 10. Summary of What Claude Does on Turn 1

1. Read `project 3 overview - connect 4 new.pdf` at the project root.
2. Look at the three model folders to understand what's there (but don't load anything yet).
3. Produce a **numbered plan** for the first chunk of work, scoped to questions 1–3.
4. Wait for my confirmation before writing any code.

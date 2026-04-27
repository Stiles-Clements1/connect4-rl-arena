# SAC training notes — Zan

These notes capture the full path from raw PG to the final SAC submission:
what we tried, what broke, what worked, and why the final run ended at
~80% mean win rate. The structure is deliberately chunked so it can be
lifted into `report/report.tex` (§3 and §4) with minimal editing.

---

## 1. Starting point — vanilla policy gradient (Q1–Q3 baseline)

The Q1–Q3 deliverable was a textbook REINFORCE loop:

- **M1:** the Stiles transformer (`Stiles Group Models/transformer_v2.keras`).
- **Loss:** `L = -mean( G_t · log π(a_t | s_t) )` with one gradient step
  per batch of discounted returns collected from self-play.
- **Opponent pool:** the six Project-1 pretrained networks, never removed,
  plus periodically-added frozen snapshots of M1.
- **Tricks:** baseline subtraction by batch-normalising returns, gradient
  clipping, entropy bonus, random warm-up moves to diversify starts.

Baseline performance against the pool plateaued around **55–60% mean
win rate** on deterministic head-to-head evaluation. Vanilla PG was
stable enough to graduate Q3, but was clearly leaving signal on the
table — especially against the strongest opponent (the Zan CNN).

---

## 2. What went wrong between PG and SAC

Three failure modes drove the method changes:

1. **High-variance gradient updates.** Connect-4 has a sparse reward
   (only ±1 at game end). REINFORCE's G_t is the full discounted return
   from game end, so every move in a long game carries the same raw
   reward, differing only by γ^k. That makes the gradient noisy: two
   moves that looked equally "good" to the agent could get opposite
   signs purely on the game's end condition.
2. **Catastrophic forgetting when the pool drifted.** Once M1 got
   stronger than most of the pool, snapshot-only self-play let bad
   habits compound. Random snapshot sampling wasn't enough — we
   needed stronger weighting toward the original pretrained networks.
3. **No credit assignment.** A blunder on move 7 that lost the game on
   move 25 was indistinguishable from the move-24 blunder. γ = 0.99
   muddles them too slowly; γ = 0.9 is punitively myopic.

These are not bugs in the PG code. They are structural limits of
REINFORCE with sparse rewards.

---

## 3. First fix attempt — Actor-Critic

We implemented an actor-critic with a value baseline and a target
critic network (DDQN-style stability trick applied to V(s) instead of
Q(s, a)).

### Why actor-critic

The critic gives us an advantage `A(s) = G − V(s)` instead of raw `G`.
That subtracts a state-dependent baseline, which *mathematically* has
zero bias but dramatically *lower variance* than `G`. The PPO option
in the AC trainer added a clipped trust-region objective on top for
extra stability.

### Why we moved off of AC

Two things showed up in long runs (200+ groups):

- **PPO's exp(log_diff) ratio occasionally blew up numerically** when
  the policy became sharp, driving the policy to NaN in a single
  batch. We added NaN guards and made vanilla AC the default, but the
  underlying instability remained if shaping + advantage + value
  regression produced a pathological batch.
- **AC is still on-policy.** Every transition is used once for one
  gradient step, then discarded. Connect-4 self-play is expensive
  (each game is 20+ moves × 2 sides × one forward pass per move) so
  sample efficiency matters a lot. On-policy training was the
  bottleneck.

AC got us to ~65% mean win rate — a real improvement over PG, but we
could feel it stalling, and the forgetting issue wasn't fully solved.

---

## 4. Final method — Soft Actor-Critic (discrete)

SAC is the genuine fusion of the two RL methods in the course syllabus:
**policy gradient on the actor side, Q-learning with a target network
on the critic side.** For discrete actions the formulation is:

- **Q-loss:**
  `MSE( Q(s, a), r + γⁿ (1 - done) · Σ_{a'} π(a'|s') [Q_target(s', a') − α log π(a'|s')] )`
- **Policy-loss:**
  `E_s [ Σ_a π(a|s) · (α log π(a|s) − Q(s, a)) ]`

Compared to AC, the three reasons SAC fit this project better:

1. **Q(s, a) per action, bootstrapped.** No more waiting for a full
   game to attribute credit. Each move's Q-value is updated from the
   next move's Q-value plus the immediate reward. Credit propagates
   one move per gradient step.
2. **Replay buffer → off-policy.** Every transition can be reused many
   times. Sample efficiency is an order of magnitude better than AC.
3. **Target network + entropy regularisation.** The target Q network
   is polyak-blended (soft update), so the bootstrap target drifts
   slowly and doesn't chase its own tail. The entropy term `α log π`
   in the policy loss prevents the policy from collapsing to a
   single column when it finds a locally-strong move.

### Architectural choices

- **Policy warm-start from the Zan CNN.** Rather than build a fresh
  network, the SAC policy head is the Zan CNN's 7-column softmax
  output (weights preserved). The Q head is brand-new (random init)
  and hooks off the Zan CNN's penultimate layer — so actor and critic
  share a trunk. This cuts "time to first win" dramatically because
  the policy starts at a ~75% supervised baseline instead of at
  random.
- **N-step Bellman bootstrap, n = 3.** 1-step TD propagates credit one
  move per update; Connect-4 games are 20+ moves long, so 1-step is
  slow. n = 3 is the sweet spot — faster credit assignment without
  the variance of full Monte-Carlo returns.
- **Horizontal mirror symmetry augmentation.** Column c and column
  6 − c are equivalent play. Every transition we collect from
  self-play also contributes its mirrored twin to the replay buffer.
  Doubles effective training data at near-zero CPU cost.
- **Reward shaping on net open-3 threats.** Each move gets a small
  `±shaping_coef · (my new threats − opp new threats)` reward on top
  of the terminal ±1. Kept low (coef = 0.03) so it couldn't swamp the
  real reward; just enough to give the critic a dense signal.

### Hyperparameters that mattered most

| Knob | Final value | Why |
|---|---|---|
| `alpha` (entropy temperature) | 0.03 | 0.1 caused un-sharpening (too much entropy pressure in late training). 0.03 kept exploration without softening the winning policy. |
| `n_step` | 3 | 1 was too slow; 5 was too high variance. |
| `tau` (target update) | 0.005 | Standard SAC default — slow enough that the target drift doesn't destabilise training. |
| `pool_originals_weight` | 3.0 | Equal weighting let snapshots dominate late; 3× on originals kept strong baseline opponents in rotation. |
| `shaping_coef` | 0.03 | Shaping disabled (0.0) → slow learning; 0.1 → policy ignored terminal reward. |
| `tactics_prob` (M2) | 0.5 | Half the games used immediate-win/block tactics for M2, half didn't. Gave M1 exposure to both rule-augmented and pure-network opponents. |
| `buffer_capacity` | 50 000 | Enough to keep roughly 1 000 games of transitions in the window. |
| `batch_size` | 128 | Stable; 64 noisier, 256 didn't help. Kept constant across updates (TF retracing rule). |

### Pool composition

The initial opponent pool excluded minimax entirely during training:

```
stiles_transformer_orig, stiles_cnn, luke_cnn, luke_transformer, zan_cnn, zan_transformer
```

Minimax was kept OUT of training because it's deterministic and would
dominate low-depth training; it is used as an **evaluation benchmark**
only. The pool grows with SAC snapshots (`SAC_snap_<id>`) every 30
groups, capped at 12 — when full, the oldest snapshot is evicted, so
the originals are always present.

---

## 5. Training run — what the numbers looked like

The final run used:

- `num_groups = 400`, `games_per_group = 32`, `updates_per_group = 16`
- Warm-start from Zan CNN policy head + fresh Q head
- Intermittent Drive-save every `checkpoint_interval = 20` groups so a
  Colab timeout couldn't lose progress

Rolling win rate vs. the training pool went through three phases:

1. **Groups 1–50 — warm-up phase.** Replay buffer was still filling
   (< `min_buffer_size`) so no SAC updates happened. The policy was
   the Zan CNN supervised baseline; win rate hovered at ~70–75%.
2. **Groups 50–150 — Q-head learning phase.** Win rate dipped to
   ~55–60% briefly as the Q head started training from scratch and
   the policy got pulled toward whatever Q said was good — even when
   Q was still wrong. Q-loss dropped from ~1.0 to ~0.1.
3. **Groups 150–400 — convergence.** Win rate climbed steadily to
   ~80% and stayed there. Entropy settled at ~1.5 (vs. ~1.95 for
   uniform over 7 columns) — confident but not fully collapsed.

### Final head-to-head eval (40 games each, random_init_moves=4)

| Opponent | Win rate | Draw | Loss |
|---|---|---|---|
| `stiles_transformer_orig` | ~88% | ~5% | ~7% |
| `zan_cnn` (supervised baseline) | ~62% | ~3% | ~35% |
| `minimax_d3` | ~80% | ~2% | ~18% |
| `random` | ~98% | ~1% | ~1% |
| **Mean vs. all** | **~80%** | — | — |

Notably, SAC *exceeded* the Zan CNN supervised baseline (~75% against
the same opponent set) — the warm-start was refined, not just reused.

---

## 6. Things that did NOT work (worth recording)

- **PPO on top of AC.** NaN-prone under sharp policies. Disabled by
  default; replaced by SAC, which fixed the underlying stability issue.
- **Training against minimax depth 5.** Tempting as a "perfect teacher"
  but (a) too slow for enough games-per-group and (b) deterministic,
  so M1 overfit to minimax's specific move preferences. Using minimax
  only for evaluation is strictly better.
- **Higher entropy temperature (α = 0.1).** Kept the policy too soft
  in late training; mean win rate topped out ~70%.
- **Disabling tactics entirely (`tactics_prob = 0.0`).** Opponents
  blundered too often; M1 didn't need to learn real Connect-4 to
  beat them, so it developed brittle local tactics that collapsed
  against calibrated minimax.
- **Checkpointing only at the end.** One Colab timeout cost 90
  minutes of progress. Solution: `checkpoint_interval = 20`.

---

## 7. How to reproduce / continue training

The code path is:

```
notebooks/sac_training.ipynb  → src/sac_trainer.py : train()
```

All hyperparameters live in `SACConfig` inside the notebook. The cell
set-up:

- **Cell 1** detects Colab vs. local and fixes up `sys.path`.
- **Cell 2** builds the initial opponent pool (pretrained-only, no minimax).
- **Cell 3** sets up benchmark opponents for in-training eval.
- **Cell 3.5** warm-starts SAC from the Zan CNN.
- **Cell 4** constructs `SACConfig` with all the tuned knobs.
- **Cell 5** runs `train(...)` — idempotent: re-running it resumes
  from `checkpoints/sac_model.keras` if one exists.
- **Cell 6/7** plot training curves and run final evaluation.

To continue training from the included checkpoint, just open the
notebook and run all cells. To start from scratch, delete
`checkpoints/sac_model.keras`, `sac_target.keras`, and `state.json`
before running.

To promote a new model as the tournament submission, copy it to
`RL models/soft_actor_critic.keras` — the commented "save finished model" cell
at the bottom of the training notebook does this for you.

---

## 8. What this proves for the report (Q7)

- **Both covered RL methods contribute.** The policy-gradient side
  (actor, SGD on log-probs) is what lets SAC improve a supervised
  baseline. The Q-learning side (target network, Bellman bootstrap,
  replay buffer) is what stabilises it and makes the fine-tuning
  sample-efficient.
- **Warm-starting from a supervised baseline dominates training from
  scratch.** RL is a fine-tuner here; the trunk features the Zan CNN
  learned by imitating MCTS are far cheaper to inherit than to
  rediscover.
- **Pool composition is as important as the loss function.** Weighting
  originals more heavily than snapshots was the single easiest fix
  for catastrophic forgetting.

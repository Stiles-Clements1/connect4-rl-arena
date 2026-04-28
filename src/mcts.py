"""
mcts.py — Monte-Carlo Tree Search agent guided by a neural-network prior.

Wraps a `ModelWrapper` (typically the trained Soft Actor-Critic submission)
so MCTS can use its policy head as the action prior `P(a|s)` and either its
Q heads or random rollouts to evaluate leaf positions. The agent exposes
the same `select_move(board, player)` interface used by every other agent
in `src/eval.py`, so it slots into `play_match` and `run_round_robin`
without further wiring.

This is a self-contained AlphaZero-style implementation:

  - **Selection.**   Walk down the tree by PUCT until reaching an
    unexpanded leaf or a terminal state.
  - **Expansion.**   Run the network once at the leaf to obtain the
    policy prior and (optionally) Q-values.
  - **Evaluation.**  Estimate the leaf's value either as
    `Σ_a π(a|s) · Q(s,a)` (`value_method="mean_q"`) or as a single
    random rollout to terminal (`value_method="rollout"`).
  - **Backup.**      Propagate the value up the path, flipping sign at
    each level (negamax-style, since perspective alternates).

All values inside MCTS are kept in the perspective of the player to move
at that node, in [-1, 1].
"""

from __future__ import annotations

import math
import random as _random
from typing import Optional

import numpy as np
import tensorflow as tf

from . import game_engine as ge
from . import model_loader as ml


# ─── Tree node ────────────────────────────────────────────────────────────────


class _MCTSNode:
    """One node of the MCTS tree.

    `value_sum` is summed in the perspective of the player whose TURN it is
    at this node.  Mean value (`Q`) from this node's POV is therefore
    `value_sum / visits`.

    From a parent node's POV, the value of taking the action that leads to
    this child is `-(value_sum / visits)` — the perspective flip across one
    ply is what `_select_child` corrects for.
    """

    __slots__ = (
        "board", "player", "prior",
        "children", "visits", "value_sum",
        "is_terminal", "terminal_value",
        "virtual_loss",
    )

    def __init__(self, board: np.ndarray, player: int, prior: float = 0.0):
        self.board: np.ndarray = board
        self.player: int = player
        self.prior: float = prior
        self.children: dict[int, "_MCTSNode"] = {}
        self.visits: int = 0
        self.value_sum: float = 0.0
        self.is_terminal: bool = False
        self.terminal_value: float = 0.0
        # Virtual loss: incremented when a parallel sim has selected this node
        # but its real result hasn't been backed up yet. PUCT treats virtual
        # visits as if they were losses, so other simulations diverge to
        # different paths instead of all stacking onto the same leaf.
        self.virtual_loss: int = 0

    def is_expanded(self) -> bool:
        return len(self.children) > 0 or self.is_terminal

    def q_value(self) -> float:
        """Mean value from THIS node's player's POV."""
        return self.value_sum / self.visits if self.visits > 0 else 0.0


# ─── Agent ────────────────────────────────────────────────────────────────────


class MCTSAgent:
    """
    MCTS agent guided by a neural-network policy prior.

    Parameters
    ----------
    wrapper : ModelWrapper
        The neural network. Its first 7-column output is treated as the
        policy prior. If the model has additional 7-column outputs (SAC's
        Q1/Q2 heads), those are used for `value_method="mean_q"`.
    n_simulations : int
        Number of MCTS rollouts per move. Higher = stronger but slower.
        Defaults to 100.
    c_puct : float
        Exploration constant in the PUCT formula. Higher = more exploration
        of low-prior moves. Standard AlphaZero values are 1.0–2.0.
    value_method : {"mean_q", "rollout"}
        How to estimate a leaf node's value.
        - "mean_q": V(s) ≈ Σ_a π(a|s) · Q(s,a) using the network's Q heads.
          Fast and low-variance, but requires the model to expose Q heads.
        - "rollout": play random moves from the leaf to terminal; return
          the outcome from the leaf's POV. Slower per simulation but works
          on policy-only networks.
    n_parallel_sims : int
        Number of MCTS simulations evaluated in a single batched forward
        pass. Defaults to 8. With sequential MCTS (n_parallel_sims=1) on a
        GPU, the per-call overhead (PCIe transfer + GPU->CPU sync to
        read policy/Q outputs) dominates the actual compute, so even a
        T4 won't beat CPU for batch=1 inference. Batching N leaves into
        one forward pass amortises that overhead and gives ~5-10x
        speedup on GPU. Selection across the N parallel paths uses
        "virtual loss" to diversify (so they don't all pick the same
        path); this is the standard AlphaZero approach. Setting this
        higher than 16 is rarely useful; setting it to 1 disables
        parallelization and gives bit-for-bit-identical-to-sequential
        behaviour at the cost of speed.
    use_tactics : bool
        If True, take an immediate winning move or block an immediate
        opponent win before invoking MCTS. Mirrors the behaviour of
        `ModelAgent` in `src.eval`. Default True.
    add_root_noise : bool
        If True, mix Dirichlet(α=0.3) noise into the root prior at weight
        0.25 (AlphaZero's default). Useful for self-play; for tournament
        play with a deterministic eval, leave False. Default False.
    name : str, optional
        Display name. Defaults to ``"mcts_n<sims>_<value_method>"``.
    """

    def __init__(
        self,
        wrapper: ml.ModelWrapper,
        n_simulations: int = 100,
        c_puct: float = 1.4,
        value_method: str = "mean_q",
        n_parallel_sims: int = 8,
        use_tactics: bool = True,
        add_root_noise: bool = False,
        name: Optional[str] = None,
    ):
        if value_method not in ("mean_q", "rollout"):
            raise ValueError(
                f"value_method must be 'mean_q' or 'rollout', got {value_method!r}"
            )
        if n_simulations < 1:
            raise ValueError(f"n_simulations must be >= 1, got {n_simulations}")
        if n_parallel_sims < 1:
            raise ValueError(
                f"n_parallel_sims must be >= 1, got {n_parallel_sims}"
            )

        self.wrapper = wrapper
        self.n_simulations = n_simulations
        self.c_puct = c_puct
        self.value_method = value_method
        self.n_parallel_sims = n_parallel_sims
        self.use_tactics = use_tactics
        self.add_root_noise = add_root_noise
        self.name = name or f"mcts_n{n_simulations}_{value_method}"

        # Pre-compile the forward pass with @tf.function so the dozens-of-
        # thousands of calls per match share one cached TF graph instead
        # of paying Python -> TF conversion overhead each time. On CPU
        # this typically saves 2-3x on inference; on GPU the win is
        # smaller but still meaningful at small batch sizes.
        @tf.function(reduce_retracing=True)
        def _compiled_forward(x):
            return wrapper.model(x, training=False)
        self._compiled_forward = _compiled_forward

        # Probe whether the network exposes Q heads (used by mean_q).
        # Cheap one-time forward pass on an empty board.
        _, q_test = self._eval_network(
            np.zeros((6, 7), dtype=np.int8), player=+1,
        )
        self.has_q_heads = q_test is not None
        if value_method == "mean_q" and not self.has_q_heads:
            raise ValueError(
                f"value_method='mean_q' requires the network to expose Q "
                f"heads (additional 7-column outputs beyond the policy), "
                f"but {wrapper.name!r} only outputs a policy. "
                f"Use value_method='rollout' instead."
            )

    # ── public ────────────────────────────────────────────────────────────────

    def select_move(self, board: np.ndarray, player: int) -> int:
        """Pick a column for `player` from the canonical board.

        Tactical shortcuts (immediate-win / immediate-block) are applied
        first if `use_tactics` is True; otherwise MCTS runs over all legal
        moves and the most-visited child of the root wins.
        """
        # Tactics — same shortcuts as ModelAgent in src/eval.py.
        if self.use_tactics:
            col = ge.winning_move(board, player)
            if col is not None:
                return col
            col = ge.blocking_move(board, player)
            if col is not None:
                return col

        # Build root and run simulations.
        root = _MCTSNode(board.copy(), player)
        self._expand_and_evaluate(root, is_root=True)

        # Batched simulations. With n_parallel_sims=1 this degenerates to
        # sequential MCTS (the standard implementation); with >1 it uses
        # virtual loss to select multiple diverging paths and evaluates
        # all their leaves in a single batched NN call -- the GPU win.
        n_remaining = self.n_simulations
        while n_remaining > 0:
            batch = min(self.n_parallel_sims, n_remaining)
            self._batched_simulate(root, batch)
            n_remaining -= batch

        # Pick the most-visited child. Ties broken by Q value.
        if not root.children:
            # Should be unreachable on a non-terminal board.
            return ge.legal_moves(board)[0]
        best_col, _ = max(
            root.children.items(),
            key=lambda kv: (kv[1].visits, kv[1].q_value()),
        )
        return best_col

    # ── search ────────────────────────────────────────────────────────────────

    def _batched_simulate(self, root: _MCTSNode, batch_size: int) -> None:
        """Run `batch_size` MCTS iterations with leaf parallelization.

        Phase 1: select `batch_size` paths through the current tree using
        PUCT + virtual loss (so they diverge to different leaves).
        Phase 2: evaluate all non-terminal leaves in ONE batched forward
        pass (the whole point of this method on GPU).
        Phase 3: back up real values along each path; remove virtual losses.

        With batch_size=1 this degenerates to sequential MCTS: virtual
        loss is added then immediately removed during the same call, and
        the search behaviour is identical to a non-batched implementation.
        """
        # ── Phase 1: Selection ──────────────────────────────────────────
        # For each parallel sim, walk the tree to a leaf and apply virtual
        # loss along the path so subsequent path selections diverge.
        paths: list[list[_MCTSNode]] = []
        leaves: list[_MCTSNode] = []
        for _ in range(batch_size):
            path: list[_MCTSNode] = [root]
            node = root
            while node.is_expanded() and not node.is_terminal:
                _, child = self._select_child(node)
                node = child
                path.append(node)
            for n in path:
                n.virtual_loss += 1
            paths.append(path)
            leaves.append(node)

        # ── Phase 2: Batched evaluation ─────────────────────────────────
        # Terminal leaves use stored values; non-terminal leaves are
        # evaluated together in a single forward pass.
        leaf_values: list[float] = [0.0] * batch_size

        eval_indices: list[int] = []
        for i, leaf in enumerate(leaves):
            if leaf.is_terminal:
                leaf_values[i] = leaf.terminal_value

        # Re-check terminal for newly-expanded leaves (the leaf might be a
        # board state that's already a win/draw -- only matters for the very
        # first sim of a freshly-built root, but cheap to handle uniformly).
        for i, leaf in enumerate(leaves):
            if leaf.is_terminal:
                continue
            done, winner = ge.is_terminal(leaf.board)
            if done:
                leaf.is_terminal = True
                leaf.terminal_value = 0.0 if winner == 0 else -1.0
                leaf_values[i] = leaf.terminal_value
            else:
                eval_indices.append(i)

        if eval_indices:
            # Stack inputs and do one batched forward pass.
            xs = np.stack([
                ml.encode_board(self.wrapper, leaves[i].board, leaves[i].player)
                for i in eval_indices
            ]).astype(np.float32)
            if hasattr(self, "_compiled_forward"):
                raw = self._compiled_forward(tf.constant(xs))
            else:
                raw = self.wrapper.model(xs, training=False)
            policies, q_values = self._parse_batch_outputs(raw, len(eval_indices))

            for batch_idx, leaf_idx in enumerate(eval_indices):
                leaf = leaves[leaf_idx]
                legal = ge.legal_moves(leaf.board)
                priors = self._legal_prior(policies[batch_idx], legal)
                # Create children (we already verified non-terminal above).
                for c in legal:
                    child_board = ge.make_move(leaf.board, c, leaf.player)
                    leaf.children[c] = _MCTSNode(
                        child_board, -leaf.player, prior=float(priors[c]),
                    )
                if self.value_method == "mean_q" and q_values is not None:
                    q = q_values[batch_idx]
                    v = float(np.sum(priors[legal] * q[legal]))
                    leaf_values[leaf_idx] = float(np.clip(v, -1.0, 1.0))
                else:
                    leaf_values[leaf_idx] = self._rollout_value(leaf)

        # ── Phase 3: Backup ─────────────────────────────────────────────
        # Remove virtual losses, add real visit + value with negamax sign-flip.
        for path, value in zip(paths, leaf_values):
            v = value
            for n in reversed(path):
                n.virtual_loss -= 1
                n.visits += 1
                n.value_sum += v
                v = -v

    @staticmethod
    def _parse_batch_outputs(
        raw, batch_size: int,
    ) -> tuple[np.ndarray, Optional[np.ndarray]]:
        """Pull (batched_policy, batched_q_or_None) out of the network's
        outputs. ``raw`` may be a single tensor or a list of tensors; we
        keep only those whose trailing dim is 7 and reshape each to
        (batch_size, 7). Following the same convention as the single-leaf
        path: first 7-col output is the policy, second/third are Q heads
        (taking element-wise min for SAC double-Q)."""
        if not isinstance(raw, (list, tuple)):
            raw = [raw]
        seven_outs: list[np.ndarray] = []
        for o in raw:
            try:
                arr = o.numpy() if hasattr(o, "numpy") else np.asarray(o)
            except Exception:
                continue
            if arr.ndim >= 2 and arr.shape[-1] == 7:
                seven_outs.append(
                    arr.reshape(batch_size, 7).astype(np.float32)
                )
        if not seven_outs:
            return (
                np.full((batch_size, 7), 1.0 / 7, dtype=np.float32),
                None,
            )
        policies = seven_outs[0]
        if len(seven_outs) >= 3:
            q = np.minimum(seven_outs[1], seven_outs[2])
        elif len(seven_outs) == 2:
            q = seven_outs[1]
        else:
            q = None
        return policies, q

    def _select_child(self, node: _MCTSNode) -> tuple[int, _MCTSNode]:
        """PUCT child selection with virtual-loss support.

        Score = -effective_q + c_puct · prior · √(N_parent_eff) / (1 + N_child_eff)

        where N_*_eff = visits + virtual_loss. The virtual-loss
        adjustment to value_sum looks counter-intuitive at first: we
        ADD virtual_loss to child.value_sum rather than subtracting.
        Reason: ``value_sum`` is stored in the CHILD's player's POV
        (negamax convention). A pending visit through this child
        represents a tentative "loss" for the player who chose to go
        here -- the PARENT. From the parent's POV this is a -1; from
        the child's POV it is a +1. Adding virtual_loss to the child's
        value_sum (in its own POV) is exactly the negamax-correct way
        to record that loss-for-parent. Parent's PUCT score for this
        child is then -effective_value_sum / effective_visits, which
        decreases as virtual_loss grows -- so other parallel sims see
        this child as less attractive and diverge to siblings, which
        is the whole point of virtual loss.
        """
        parent_eff = node.visits + node.virtual_loss
        sqrt_total = math.sqrt(max(parent_eff, 1))
        best_score = -float("inf")
        best_pair: tuple[int, _MCTSNode] | None = None

        for col, child in node.children.items():
            child_eff = child.visits + child.virtual_loss
            if child_eff > 0:
                # +virtual_loss in the child's POV = -virtual_loss in the
                # parent's POV (negamax). See docstring above for why.
                effective_value_sum = child.value_sum + child.virtual_loss
                q_parent_pov = -effective_value_sum / child_eff
            else:
                q_parent_pov = 0.0
            u = self.c_puct * child.prior * sqrt_total / (1 + child_eff)
            score = q_parent_pov + u
            if score > best_score:
                best_score = score
                best_pair = (col, child)
        # `best_pair` is None only if `node.children` was empty, which we've
        # already guarded against in the simulation loop.
        assert best_pair is not None
        return best_pair

    def _expand_and_evaluate(self, node: _MCTSNode, *, is_root: bool) -> float:
        """Expand `node` (set its children's priors) and return the value at
        this node from `node.player`'s POV.

        Handles three cases:
          1. The node is already terminal — set `terminal_value`, return it.
          2. The node is non-terminal — run the network, create children
             with priors, and return either a Q-mean estimate or a rollout
             estimate (per `self.value_method`).
        """
        # Terminal check.
        done, winner = ge.is_terminal(node.board)
        if done:
            node.is_terminal = True
            if winner == 0:
                node.terminal_value = 0.0
            else:
                # The board is terminal AND it's `node.player`'s turn. That
                # means the OTHER player just made the move that won, so
                # `node.player` has lost.
                node.terminal_value = -1.0
            return node.terminal_value

        # Network evaluation.
        policy_raw, q_values = self._eval_network(node.board, node.player)
        legal = ge.legal_moves(node.board)

        # Mask + normalise to a prior over legal moves.
        priors = self._legal_prior(policy_raw, legal)

        # Optional Dirichlet noise at the root (AlphaZero default).
        if is_root and self.add_root_noise and len(legal) > 0:
            noise = np.random.dirichlet([0.3] * len(legal))
            for i, c in enumerate(legal):
                priors[c] = 0.75 * priors[c] + 0.25 * noise[i]

        # Create children.
        for c in legal:
            child_board = ge.make_move(node.board, c, node.player)
            node.children[c] = _MCTSNode(
                child_board, -node.player, prior=float(priors[c]),
            )

        # Estimate value at this node from node.player's POV.
        if self.value_method == "mean_q":
            # V(s) = Σ_a π(a|s) Q(s, a) over legal moves only.
            v = float(np.sum(priors[legal] * q_values[legal]))
            v = float(np.clip(v, -1.0, 1.0))
        else:  # rollout
            v = self._rollout_value(node)
        return v

    def _rollout_value(self, node: _MCTSNode) -> float:
        """Random playout from `node` to terminal; return ±1/0 from
        `node.player`'s POV."""
        board = node.board.copy()
        cur = node.player
        for _ in range(64):  # safety cap; Connect-4 ends within 42 moves
            done, winner = ge.is_terminal(board)
            if done:
                if winner == 0:
                    return 0.0
                return 1.0 if winner == node.player else -1.0
            legal = ge.legal_moves(board)
            if not legal:
                return 0.0
            col = _random.choice(legal)
            board = ge.make_move(board, col, cur)
            cur = -cur
        return 0.0  # shouldn't reach here

    # ── network plumbing ──────────────────────────────────────────────────────

    def _eval_network(
        self, board: np.ndarray, player: int,
    ) -> tuple[np.ndarray, Optional[np.ndarray]]:
        """Single forward pass; return ``(policy, q_values_or_None)``.

        - ``policy``: shape (7,), the FIRST 7-column network output.
        - ``q_values``: shape (7,), per-action Q values from later 7-column
          outputs. If the network has 3+ such outputs (SAC double-Q), this
          is ``min(Q1, Q2)`` per the standard SAC convention. If only one
          additional output exists, that is used directly. ``None`` if the
          network has no Q heads.
        """
        x = ml.encode_board(self.wrapper, board, player)[np.newaxis].astype(
            np.float32,
        )
        # Use the compiled graph if it exists (it does after __init__);
        # the very first probe call from __init__ goes through the raw
        # Keras model because _compiled_forward isn't bound yet.
        if hasattr(self, "_compiled_forward"):
            raw = self._compiled_forward(tf.constant(x))
        else:
            raw = self.wrapper.model(x, training=False)

        if not isinstance(raw, (list, tuple)):
            raw = [raw]

        seven_outs: list[np.ndarray] = []
        for o in raw:
            try:
                arr = o.numpy() if hasattr(o, "numpy") else np.asarray(o)
            except Exception:
                continue
            if arr.ndim >= 1 and arr.shape[-1] == 7:
                seven_outs.append(arr.flatten()[:7].astype(np.float32))

        if not seven_outs:
            # No 7-column output at all — degenerate model. Fall back to
            # uniform priors so the agent doesn't crash.
            return np.full(7, 1.0 / 7, dtype=np.float32), None

        policy = seven_outs[0]
        if len(seven_outs) >= 3:
            # SAC-style double-Q — take element-wise min, the conservative
            # estimate that prevents value over-estimation.
            q = np.minimum(seven_outs[1], seven_outs[2])
        elif len(seven_outs) == 2:
            q = seven_outs[1]
        else:
            q = None
        return policy, q

    @staticmethod
    def _legal_prior(policy_raw: np.ndarray, legal: list[int]) -> np.ndarray:
        """Mask illegal columns and renormalise.

        Robust to whether the policy head outputs raw logits or already-
        softmaxed probabilities: we clip negatives to zero (which only
        matters for raw logits) and renormalise the remaining mass over
        legal moves.
        """
        priors = np.zeros(7, dtype=np.float32)
        legal_arr = np.maximum(policy_raw[legal], 0.0)
        s = float(legal_arr.sum())
        if s > 1e-8:
            priors[legal] = legal_arr / s
        else:
            priors[legal] = 1.0 / len(legal)
        return priors

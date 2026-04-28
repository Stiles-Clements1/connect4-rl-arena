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

        self.wrapper = wrapper
        self.n_simulations = n_simulations
        self.c_puct = c_puct
        self.value_method = value_method
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

        for _ in range(self.n_simulations):
            self._simulate(root)

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

    def _simulate(self, root: _MCTSNode) -> None:
        """One MCTS iteration: select → expand+evaluate → backup."""
        path: list[_MCTSNode] = [root]
        node = root

        # 1. Selection — walk down using PUCT until unexpanded or terminal.
        while node.is_expanded() and not node.is_terminal:
            _, child = self._select_child(node)
            node = child
            path.append(node)

        # 2. Expansion + leaf evaluation.
        if node.is_terminal:
            leaf_value = node.terminal_value  # already in node.player's POV
        else:
            leaf_value = self._expand_and_evaluate(node, is_root=False)

        # 3. Backup. `leaf_value` is in path[-1].player's POV. Going up, the
        # POV flips at every level.
        v = leaf_value
        for n in reversed(path):
            n.visits += 1
            n.value_sum += v
            v = -v

    def _select_child(self, node: _MCTSNode) -> tuple[int, _MCTSNode]:
        """PUCT child selection.

        Score = -child.q_value() + c_puct · prior · √(N_parent) / (1 + N_child)

        The negation on `child.q_value()` flips child's POV into the
        parent's POV (which is what we are choosing for).
        """
        sqrt_total = math.sqrt(max(node.visits, 1))
        best_score = -float("inf")
        best_pair: tuple[int, _MCTSNode] | None = None

        for col, child in node.children.items():
            # Q from PARENT's POV is the negation of CHILD's recorded Q.
            q_parent_pov = -child.q_value() if child.visits > 0 else 0.0
            u = self.c_puct * child.prior * sqrt_total / (1 + child.visits)
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

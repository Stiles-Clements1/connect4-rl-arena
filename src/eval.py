"""
eval.py — Head-to-head and round-robin evaluation of Connect-4 agents.

This module plays agents against each other for a fixed number of games,
alternating first-player and respecting each agent's configuration (greedy
vs. stochastic move selection, tactical overrides on/off), and returns
clear summary statistics.

The intent is to answer:  "Which of our models is actually strongest?"
Training-time win rates are not directly comparable across agents (each
agent was trained against a different opponent mix), so head-to-head play
is what we use for the Q5 comparison and for tournament model selection.
"""

import itertools
import json
import random as _random
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional, Union

import numpy as np

try:
    from tqdm.auto import tqdm
except ImportError:
    # tqdm is optional; fall back to a no-op wrapper if it's not installed
    def tqdm(iterable, **kwargs):
        return iterable

from . import game_engine as ge
from . import model_loader as ml
from .model_loader import ModelWrapper


# ─────────────────────────────────────────────────────────────────────────────
# Shared move-selection helper
# ─────────────────────────────────────────────────────────────────────────────

def _pick_column(scores: np.ndarray, legal: list, greedy: bool) -> int:
    """
    Pick a column in [0, 6] from a 7-element `scores` array, restricted to
    `legal`. If `greedy`, returns `argmax` over legal columns. Otherwise
    returns a sample from the masked, renormalized distribution; if the
    distribution is near-zero everywhere, falls back to uniform over legal
    columns.
    """
    if greedy:
        masked = np.full(7, -np.inf, dtype=np.float32)
        masked[legal] = scores[legal]
        return int(np.argmax(masked))

    masked = np.zeros(7, dtype=np.float32)
    masked[legal] = scores[legal]
    total = masked.sum()
    if total > 1e-8:
        masked /= total
    else:
        masked[legal] = 1.0 / len(legal)
    return int(np.random.choice(7, p=masked))


# ─────────────────────────────────────────────────────────────────────────────
# Agents
# ─────────────────────────────────────────────────────────────────────────────

class ModelAgent:
    """
    Wraps a ModelWrapper so it can play a game.

    Move selection order on each turn:
      1. If `use_tactics` and an immediate winning move exists, play it.
      2. If `use_tactics` and the opponent has an immediate winning threat,
         block it.
      3. Otherwise consult the model:
           - `greedy=True`  -> argmax over legal columns of the raw model output
           - `greedy=False` -> sample stochastically from the masked softmax

    The tactical overrides (1, 2) only ever play legal moves -- they are not
    "cheats". Any competent Connect-4 player takes a one-move win or blocks a
    one-move loss. We leave them on by default because that is how the agent
    will behave at tournament time.
    """

    def __init__(
        self,
        wrapper: ModelWrapper,
        name: Optional[str] = None,
        greedy: bool = True,
        use_tactics: bool = True,
    ):
        self.wrapper     = wrapper
        self.name        = name if name is not None else wrapper.name
        self.greedy      = greedy
        self.use_tactics = use_tactics

    def select_move(self, board: np.ndarray, player: int) -> int:
        # 1-2. Tactical overrides
        if self.use_tactics:
            col = ge.winning_move(board, player)
            if col is not None:
                return col
            col = ge.blocking_move(board, player)
            if col is not None:
                return col

        # 3. Model-based selection
        legal  = ge.legal_moves(board)
        scores = ml.predict_probs(self.wrapper, board, player)   # (7,)
        return _pick_column(scores, legal, self.greedy)


class RandomAgent:
    """
    Uniformly random legal move. Baseline "floor" opponent against which
    every other agent should win decisively.
    """

    def __init__(self, name: str = "random"):
        self.name = name

    def select_move(self, board: np.ndarray, player: int) -> int:
        return int(_random.choice(ge.legal_moves(board)))


# ─────────────────────────────────────────────────────────────────────────────
# MinimaxAgent — alpha-beta search over a 4-window heuristic
# ─────────────────────────────────────────────────────────────────────────────
#
# A deterministic, non-neural Connect-4 agent with a tunable strength knob
# (search depth). Useful as:
#   - a calibrated baseline (depth=1 barely tactical, depth=3 sees short
#     threats, depth=5 strong, depth=7+ near-optimal);
#   - a potential tournament submission (a tuned depth-5 alpha-beta is
#     competitive with neural-net Connect-4 agents).
# The agent is fully deterministic for the same (board, player): ties are
# broken by a centre-preferring column order.

# Weights for 4-cell windows that sum to a board score from `player`'s view.
# Classical values for Connect-4 heuristics.
_MM_W_WIN_LINE   = 10_000   # four in a row (mostly unreachable at a leaf)
_MM_W_THREE_LINE = 50       # three of player's + one empty
_MM_W_TWO_LINE   = 5        # two of player's + two empty
_MM_W_CENTER_COL = 3        # bonus per piece in the centre column (col 3)

# Centre-out column order — tightens alpha-beta bounds earlier in the search.
_MM_COLUMN_ORDER = [3, 2, 4, 1, 5, 0, 6]

# Value safely larger than any heuristic score the position can produce.
# Terminal outcomes are scored ±(_MM_INF + depth_remaining) so the search
# prefers faster wins and delays losses.
_MM_INF = 1_000_000


def _mm_score_window(window: np.ndarray, player: int) -> int:
    """Score a 4-cell slice from `player`'s perspective."""
    me    = int(np.sum(window == player))
    opp   = int(np.sum(window == -player))
    empty = 4 - me - opp

    if me == 4:
        return _MM_W_WIN_LINE
    if opp == 4:
        return -_MM_W_WIN_LINE
    if me == 3 and empty == 1:
        return _MM_W_THREE_LINE
    if opp == 3 and empty == 1:
        return -_MM_W_THREE_LINE
    if me == 2 and empty == 2:
        return _MM_W_TWO_LINE
    if opp == 2 and empty == 2:
        return -_MM_W_TWO_LINE
    return 0


# Cache of (board_bytes, player) -> heuristic score. Connect-4 has heavy
# transposition; on deep searches the same position is often re-evaluated
# dozens of times. Capped to avoid unbounded memory growth.
_MM_HEURISTIC_CACHE: dict = {}
_MM_CACHE_CAP      = 200_000


def _mm_heuristic(board: np.ndarray, player: int) -> int:
    """
    Static evaluation of a non-terminal position from `player`'s perspective.
    Sums the window-score over all 4-cell horizontal, vertical, and diagonal
    slices, plus a small bonus for owning centre-column cells.

    Memoized on (board.tobytes(), player). np.diagonal is used for the
    diagonal windows instead of a per-cell list comprehension — together
    these give roughly a 3-5x speedup at depth 5.
    """
    key = (board.tobytes(), player)
    cached = _MM_HEURISTIC_CACHE.get(key)
    if cached is not None:
        return cached

    score = 0

    # Horizontal
    for r in range(ge.ROWS):
        for c in range(ge.COLS - 3):
            score += _mm_score_window(board[r, c:c+4], player)
    # Vertical
    for r in range(ge.ROWS - 3):
        for c in range(ge.COLS):
            score += _mm_score_window(board[r:r+4, c], player)
    # Down-right diagonals: np.diagonal of a 4x4 subboard
    for r in range(ge.ROWS - 3):
        for c in range(ge.COLS - 3):
            score += _mm_score_window(np.diagonal(board[r:r+4, c:c+4]), player)
    # Down-left diagonals: np.diagonal on a flipped 4x4 subboard
    for r in range(ge.ROWS - 3):
        for c in range(3, ge.COLS):
            score += _mm_score_window(
                np.diagonal(np.fliplr(board[r:r+4, c-3:c+1])), player
            )

    # Small centre-column bonus
    score += int(np.sum(board[:, 3] == player)) * _MM_W_CENTER_COL

    # Cache (with a simple capped-size eviction: drop everything when full)
    if len(_MM_HEURISTIC_CACHE) >= _MM_CACHE_CAP:
        _MM_HEURISTIC_CACHE.clear()
    _MM_HEURISTIC_CACHE[key] = score
    return score


def _mm_alphabeta(
    board: np.ndarray,
    depth: int,
    alpha: int,
    beta: int,
    maximizing_player: int,
    current_player: int,
):
    """
    Standard alpha-beta search. Returns (score_for_maximizing_player, column).
    `depth` is the remaining depth; `current_player` is who moves next.
    """
    done, winner = ge.is_terminal(board)
    if done:
        if winner == maximizing_player:
            return _MM_INF + depth, None
        if winner == -maximizing_player:
            return -_MM_INF - depth, None
        return 0, None

    if depth == 0:
        return _mm_heuristic(board, maximizing_player), None

    legal         = ge.legal_moves(board)
    legal_ordered = [c for c in _MM_COLUMN_ORDER if c in legal]
    best_col      = legal_ordered[0]

    if current_player == maximizing_player:
        value = -_MM_INF
        for col in legal_ordered:
            child = ge.make_move(board, col, current_player)
            score, _ = _mm_alphabeta(
                child, depth - 1, alpha, beta,
                maximizing_player, -current_player,
            )
            if score > value:
                value, best_col = score, col
            alpha = max(alpha, value)
            if alpha >= beta:
                break
        return value, best_col

    value = _MM_INF
    for col in legal_ordered:
        child = ge.make_move(board, col, current_player)
        score, _ = _mm_alphabeta(
            child, depth - 1, alpha, beta,
            maximizing_player, -current_player,
        )
        if score < value:
            value, best_col = score, col
        beta = min(beta, value)
        if alpha >= beta:
            break
    return value, best_col


class MinimaxAgent:
    """
    Deterministic alpha-beta minimax Connect-4 agent with a configurable
    search depth. Slots into play_match / play_match_parallel / run_round_robin
    the same way ModelAgent and RandomAgent do.

    Depth recommendations:
      - depth=1   barely tactical (one-move lookahead only)
      - depth=3   sees 2-3 ply threats; beats random easily
      - depth=5   strong; beats most humans
      - depth=7+  near-optimal, but hundreds of ms per move on CPU

    Tactical overrides (immediate-win / immediate-block) are NOT applied
    separately — the search already sees those at depth 1 and deeper.
    """

    def __init__(self, depth: int, name: Optional[str] = None):
        if depth < 1:
            raise ValueError(f"depth must be >= 1, got {depth}")
        self.depth = depth
        self.name  = name if name is not None else f"minimax_d{depth}"

    def select_move(self, board: np.ndarray, player: int) -> int:
        _, col = _mm_alphabeta(
            board, self.depth,
            alpha=-_MM_INF, beta=_MM_INF,
            maximizing_player=player,
            current_player=player,
        )
        if col is None:
            # Non-terminal boards always yield a move; this is a pure safety
            # net for a bug or a fully-full board reaching the search.
            return ge.legal_moves(board)[0]
        return col


# ─────────────────────────────────────────────────────────────────────────────
# Match result container
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class MatchResult:
    """Aggregated outcome of an N-game match between two agents."""
    name_a:  str
    name_b:  str
    n_games: int

    # Overall counts
    a_wins: int = 0
    b_wins: int = 0
    draws:  int = 0

    # Counts broken out by who moved first (to check first-player advantage)
    a_first_wins:   int = 0
    a_first_losses: int = 0
    a_first_draws:  int = 0
    b_first_wins:   int = 0
    b_first_losses: int = 0
    b_first_draws:  int = 0

    # For average game length
    total_moves: int = 0

    @property
    def a_win_rate(self) -> float:
        return self.a_wins / self.n_games if self.n_games else 0.0

    @property
    def b_win_rate(self) -> float:
        return self.b_wins / self.n_games if self.n_games else 0.0

    @property
    def draw_rate(self) -> float:
        return self.draws / self.n_games if self.n_games else 0.0

    @property
    def avg_length(self) -> float:
        return self.total_moves / self.n_games if self.n_games else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Single game
# ─────────────────────────────────────────────────────────────────────────────

def play_single_game(
    agent_a,
    agent_b,
    first_player: str = "a",
    random_init_moves: int = 0,
) -> dict:
    """
    Play one full game between `agent_a` and `agent_b`.

    Parameters
    ----------
    first_player : {"a", "b"}
        Which agent plays as +1 (the side that moves first on an empty board).
    random_init_moves : int
        If > 0, apply this many uniformly random legal moves before either
        agent selects a move. Used to diversify starting positions; 0 by
        default so each head-to-head game starts from the empty board.

    Returns
    -------
    dict with keys:
        'winner' : 'a', 'b', or None (draw)
        'length' : total number of moves played (including any warm-up moves)
    """
    assert first_player in ("a", "b"), "first_player must be 'a' or 'b'"

    board = np.zeros((ge.ROWS, ge.COLS), dtype=np.int8)

    a_player = +1 if first_player == "a" else -1
    b_player = -a_player

    # Optional warm-up (rare — used mainly for stress-test scenarios)
    next_player = +1
    if random_init_moves > 0:
        board, next_player = ge.random_moves(board, random_init_moves, +1)
        if next_player is None:
            return {"winner": None, "length": random_init_moves}

    move_count = 0
    while True:
        if next_player == a_player:
            col = agent_a.select_move(board, a_player)
        else:
            col = agent_b.select_move(board, b_player)

        board = ge.make_move(board, col, next_player)
        move_count += 1

        done, winner = ge.is_terminal(board)
        if done:
            if winner == a_player:
                return {"winner": "a", "length": move_count}
            if winner == b_player:
                return {"winner": "b", "length": move_count}
            return {"winner": None, "length": move_count}

        next_player = -next_player


# ─────────────────────────────────────────────────────────────────────────────
# Match (N games, alternating first-player)
# ─────────────────────────────────────────────────────────────────────────────

def play_match(
    agent_a,
    agent_b,
    n_games: int = 100,
    random_init_moves: int = 0,
    progress: bool = True,
) -> MatchResult:
    """
    Play `n_games` games between two agents. Half the games have `agent_a`
    first; the other half have `agent_b` first. The two halves are
    interleaved at random so a progress bar is evenly informative.

    Returns a MatchResult with aggregate counts plus first-player breakdown.
    """
    half = n_games // 2
    schedule = ["a"] * half + ["b"] * (n_games - half)
    _random.shuffle(schedule)

    r = MatchResult(name_a=agent_a.name, name_b=agent_b.name, n_games=n_games)

    pbar = tqdm(
        schedule,
        desc=f"{agent_a.name[:18]} vs {agent_b.name[:18]}",
        disable=not progress,
        leave=False,
    )
    for first in pbar:
        outcome = play_single_game(
            agent_a, agent_b,
            first_player=first,
            random_init_moves=random_init_moves,
        )
        r.total_moves += outcome["length"]

        winner = outcome["winner"]
        if winner == "a":
            r.a_wins += 1
            if first == "a": r.a_first_wins   += 1
            else:            r.b_first_losses += 1
        elif winner == "b":
            r.b_wins += 1
            if first == "a": r.a_first_losses += 1
            else:            r.b_first_wins   += 1
        else:  # draw
            r.draws += 1
            if first == "a": r.a_first_draws  += 1
            else:            r.b_first_draws  += 1

    return r


# ─────────────────────────────────────────────────────────────────────────────
# GPU-batched match: all games advance in lockstep, one forward pass per turn
# ─────────────────────────────────────────────────────────────────────────────

def _apply_move(i, col, player, boards, lengths, done, winner, next_player):
    """Apply one move to game `i` and update its status in the parallel arrays."""
    boards[i] = ge.make_move(boards[i], col, player)
    lengths[i] += 1
    is_done, win = ge.is_terminal(boards[i])
    if is_done:
        done[i], winner[i] = True, win
    else:
        next_player[i] = -player


def _resolve_turn_batched(agent, indices, boards, agent_player,
                          done, winner, lengths, next_player):
    """
    Advance every active game in `indices` by one half-move for `agent`.

    Tactical overrides (winning_move / blocking_move) are resolved per-game
    in Python (cheap). Everything that still needs the model is collected
    into ONE batched forward pass — that's where the GPU speedup comes from.
    """
    if not indices:
        return

    # 1. Per-game tactical overrides. Games resolved by tactics skip the batch.
    need_model = []
    if isinstance(agent, ModelAgent) and agent.use_tactics:
        for i in indices:
            p = agent_player[i]
            col = ge.winning_move(boards[i], p)
            if col is None:
                col = ge.blocking_move(boards[i], p)
            if col is not None:
                _apply_move(i, col, p, boards, lengths, done, winner, next_player)
            else:
                need_model.append(i)
    else:
        need_model = list(indices)

    if not need_model:
        return

    # 2. Batched neural-net inference for every remaining game
    if isinstance(agent, ModelAgent):
        xs = np.stack([
            ml.encode_board(agent.wrapper, boards[i], agent_player[i])
            for i in need_model
        ])
        raw = agent.wrapper.model(xs, training=False)
        raw = (raw[0] if isinstance(raw, (list, tuple)) else raw).numpy()  # (B, 7)

        for k, i in enumerate(need_model):
            p     = agent_player[i]
            legal = ge.legal_moves(boards[i])
            col   = _pick_column(raw[k], legal, agent.greedy)
            _apply_move(i, col, p, boards, lengths, done, winner, next_player)

    elif isinstance(agent, RandomAgent):
        # No neural net — per-game random legal move, still cheap Python
        for i in need_model:
            p = agent_player[i]
            col = int(_random.choice(ge.legal_moves(boards[i])))
            _apply_move(i, col, p, boards, lengths, done, winner, next_player)

    else:
        # Unknown agent type: fall back to its .select_move, one at a time
        for i in need_model:
            p   = agent_player[i]
            col = agent.select_move(boards[i], p)
            _apply_move(i, col, p, boards, lengths, done, winner, next_player)


def play_match_parallel(
    agent_a,
    agent_b,
    n_games: int = 100,
    random_init_moves: int = 0,
    progress: bool = True,
) -> MatchResult:
    """
    GPU-batched equivalent of `play_match`. All `n_games` games run
    concurrently: at each half-move we partition the active games by whose
    turn it is, then run at most ONE batched forward pass per agent instead
    of `n_games` separate batch-of-1 calls. Materially faster on A100 / T4;
    on CPU the two are roughly equivalent (no batching advantage).

    Identical signature and return type to `play_match`, so it is a drop-in
    replacement.
    """
    # Interleaved first-player schedule, matching play_match's distribution
    half = n_games // 2
    schedule = ["a"] * half + ["b"] * (n_games - half)
    _random.shuffle(schedule)

    # Per-game state, initialised in parallel
    boards       = [np.zeros((ge.ROWS, ge.COLS), dtype=np.int8) for _ in range(n_games)]
    a_player     = [+1 if s == "a" else -1 for s in schedule]
    b_player     = [-p for p in a_player]
    next_player  = [+1] * n_games
    done         = [False] * n_games
    winner       = [None] * n_games
    lengths      = [0] * n_games

    # Optional random warm-up (per-game, sequential — cheap, no NN calls)
    if random_init_moves > 0:
        for i in range(n_games):
            boards[i], np_after = ge.random_moves(boards[i], random_init_moves, +1)
            lengths[i] = random_init_moves
            if np_after is None:
                done[i] = True
            else:
                next_player[i] = np_after

    pbar = tqdm(
        total=n_games,
        desc=f"{agent_a.name[:18]} vs {agent_b.name[:18]}",
        disable=not progress,
        leave=False,
    )
    pbar.update(sum(1 for d in done if d))  # any games ended in warm-up

    # Main loop: every iteration advances all active games by one half-move
    while not all(done):
        a_turn, b_turn = [], []
        for i in range(n_games):
            if done[i]:
                continue
            if next_player[i] == a_player[i]:
                a_turn.append(i)
            else:
                b_turn.append(i)

        _resolve_turn_batched(agent_a, a_turn, boards, a_player,
                              done, winner, lengths, next_player)
        _resolve_turn_batched(agent_b, b_turn, boards, b_player,
                              done, winner, lengths, next_player)

        pbar.n = sum(1 for d in done if d)
        pbar.refresh()
    pbar.close()

    # Aggregate into a MatchResult, identical structure to play_match
    r = MatchResult(name_a=agent_a.name, name_b=agent_b.name, n_games=n_games)
    for i in range(n_games):
        r.total_moves += lengths[i]
        w = winner[i]
        if w == a_player[i]:
            r.a_wins += 1
            if schedule[i] == "a": r.a_first_wins   += 1
            else:                  r.b_first_losses += 1
        elif w == b_player[i]:
            r.b_wins += 1
            if schedule[i] == "a": r.a_first_losses += 1
            else:                  r.b_first_wins   += 1
        else:
            r.draws += 1
            if schedule[i] == "a": r.a_first_draws += 1
            else:                  r.b_first_draws += 1
    return r


# ─────────────────────────────────────────────────────────────────────────────
# Round-robin (every pair plays every other pair)
# ─────────────────────────────────────────────────────────────────────────────

def run_round_robin(
    agents: dict,
    n_games: int = 100,
    random_init_moves: int = 0,
    parallel: bool = False,
) -> dict:
    """
    Every unordered pair of agents in `agents` plays `n_games` games.

    Parameters
    ----------
    agents : dict[str, Agent]
        Name-keyed dictionary of agents to include in the tournament.
    n_games, random_init_moves
        Forwarded to the match function.
    parallel : bool
        If True, use play_match_parallel (GPU-batched, all `n_games` games
        advance in lockstep with one forward pass per turn). If False, use
        the sequential play_match. On A100 the parallel version is roughly
        10-30x faster.

    Returns
    -------
    dict keyed by (name_a, name_b) tuples, mapping to MatchResult.
    Use `round_robin_to_dataframe(results, list(agents))` to get a grid.
    """
    pairs = list(itertools.combinations(agents.keys(), 2))
    results = {}

    match_fn = play_match_parallel if parallel else play_match

    # progress=True on the inner match so users see games ticking within a
    # pair; leave=False on each match keeps the outer round-robin bar as the
    # persistent progress indicator once the pair finishes.
    for name_a, name_b in tqdm(pairs, desc="Round-robin", leave=True):
        r = match_fn(
            agents[name_a], agents[name_b],
            n_games=n_games,
            random_init_moves=random_init_moves,
            progress=True,
        )
        results[(name_a, name_b)] = r

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Formatting
# ─────────────────────────────────────────────────────────────────────────────

def format_result(r: MatchResult) -> str:
    """Pretty-print a single MatchResult to a multi-line string."""
    width_name = max(len(r.name_a), len(r.name_b), len("draws"))
    w = width_name + 2

    lines = [
        f"{r.name_a} vs {r.name_b}  ({r.n_games} games)",
        "-" * 62,
        f"  {r.name_a:<{w}s}  {r.a_wins:4d}  ({r.a_win_rate:6.1%})",
        f"  {r.name_b:<{w}s}  {r.b_wins:4d}  ({r.b_win_rate:6.1%})",
        f"  {'draws':<{w}s}  {r.draws:4d}  ({r.draw_rate:6.1%})",
        "",
        f"  avg game length: {r.avg_length:.1f} moves",
        "",
        f"  {r.name_a} as first player:  "
        f"{r.a_first_wins}W / {r.a_first_losses}L / {r.a_first_draws}D",
        f"  {r.name_b} as first player:  "
        f"{r.b_first_wins}W / {r.b_first_losses}L / {r.b_first_draws}D",
    ]
    return "\n".join(lines)


def round_robin_to_dataframe(results: dict, names: list):
    """
    Turn a round-robin results dict into a pandas DataFrame of row-agent
    win rates. Cell (i, j) is the win rate of agent i when playing agent j.
    The diagonal is filled with NaN (an agent does not play itself).
    """
    import pandas as pd

    df = pd.DataFrame(np.nan, index=names, columns=names, dtype=float)
    for (a, b), r in results.items():
        df.loc[a, b] = r.a_win_rate
        df.loc[b, a] = r.b_win_rate
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Persistence helpers — save evaluation runs so results are not lost between
# Colab sessions and can be cited / plotted / pasted in the report
# ─────────────────────────────────────────────────────────────────────────────

def _match_to_dict(r: MatchResult) -> dict:
    """JSON-serializable dict for a MatchResult, with derived rates pre-computed."""
    d = asdict(r)
    d["a_win_rate"] = r.a_win_rate
    d["b_win_rate"] = r.b_win_rate
    d["draw_rate"]  = r.draw_rate
    d["avg_length"] = r.avg_length
    return d


def save_results_json(
    results: Union[MatchResult, dict],
    path: Optional[Union[str, Path]] = None,
    tag: str = "",
    metadata: Optional[dict] = None,
) -> Path:
    """
    Save one or more MatchResult objects to a JSON file.

    Parameters
    ----------
    results : MatchResult | dict[tuple, MatchResult]
        Either a single match (from play_match / play_match_parallel) or
        the full round-robin dict (from run_round_robin).
    path : str | Path, optional
        Output path. If None, defaults to
        <repo>/logs/eval_<timestamp>[_<tag>].json
    tag : str, optional
        Short label mixed into the default filename. Example: "all_agents_100g".
    metadata : dict, optional
        Free-form extras stored alongside the matches (hardware, notebook run,
        any hyperparameters that matter for the report).

    Returns the path written to.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if path is None:
        from . import config as _cfg
        fname = f"eval_{timestamp}" + (f"_{tag}" if tag else "") + ".json"
        path  = _cfg.LOG_DIR / fname
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Normalize into a list of match dicts
    if isinstance(results, MatchResult):
        matches = [_match_to_dict(results)]
    else:
        matches = [_match_to_dict(r) for r in results.values()]

    payload = {
        "timestamp": timestamp,
        "n_matches": len(matches),
        "metadata":  metadata or {},
        "matches":   matches,
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    return path


def save_win_rate_heatmap(
    df,
    path: Optional[Union[str, Path]] = None,
    title: Optional[str] = None,
    tag: str = "",
    figsize: tuple = (10, 8),
) -> Path:
    """
    Save a round-robin win-rate DataFrame as a heatmap PNG.

    Uses matplotlib only (no seaborn dependency). Cell values are the
    row agent's win rate when playing the column agent, shown as percentages.
    Diagonal cells are rendered neutral (agent does not play itself).

    Parameters
    ----------
    df : pandas.DataFrame
        The output of round_robin_to_dataframe().
    path : str | Path, optional
        Output PNG path. If None, defaults to
        <repo>/report/figures/win_rate_matrix_<timestamp>[_<tag>].png so the
        image is ready to drop into the Q7 report.

    Returns the path written to.
    """
    import matplotlib.pyplot as plt

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if path is None:
        from . import config as _cfg
        figures_dir = _cfg.ROOT / "report" / "figures"
        fname = f"win_rate_matrix_{timestamp}" + (f"_{tag}" if tag else "") + ".png"
        path  = figures_dir / fname
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    data_pct = (df.values * 100)
    display  = np.where(np.isnan(data_pct), 50.0, data_pct)  # neutral for NaN diagonal

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(display, cmap="RdYlGn", vmin=0, vmax=100, aspect="auto")

    ax.set_xticks(np.arange(len(df.columns)))
    ax.set_yticks(np.arange(len(df.index)))
    ax.set_xticklabels(df.columns, rotation=45, ha="right")
    ax.set_yticklabels(df.index)

    # Annotate every cell with the numeric win rate (or em-dash on the diagonal)
    for i in range(len(df.index)):
        for j in range(len(df.columns)):
            val = df.iloc[i, j]
            text = "-" if np.isnan(val) else f"{val * 100:.0f}%"
            ax.text(j, i, text, ha="center", va="center", color="black", fontsize=9)

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Row agent win rate vs column agent (%)")

    ax.set_title(title or "Round-robin win rates (row vs column)")
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path

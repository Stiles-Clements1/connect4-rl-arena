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
import random as _random
from dataclasses import dataclass
from typing import Optional

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
        legal = ge.legal_moves(board)
        raw   = ml.predict_probs(self.wrapper, board, player)   # (7,)

        if self.greedy:
            legal_scores = np.full(7, -np.inf, dtype=np.float32)
            legal_scores[legal] = raw[legal]
            return int(np.argmax(legal_scores))

        # Stochastic sampling (mirrors training-time behaviour)
        masked = np.zeros(7, dtype=np.float32)
        masked[legal] = raw[legal]
        total = masked.sum()
        if total <= 1e-8:
            masked[legal] = 1.0 / len(legal)
        else:
            masked /= total
        return int(np.random.choice(7, p=masked))


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
            legal  = ge.legal_moves(boards[i])
            p      = agent_player[i]
            scores = raw[k]
            if agent.greedy:
                masked = np.full(7, -np.inf, dtype=np.float32)
                masked[legal] = scores[legal]
                col = int(np.argmax(masked))
            else:
                m = np.zeros(7, dtype=np.float32)
                m[legal] = scores[legal]
                s = m.sum()
                m = m / s if s > 1e-8 else (np.eye(1, 7, k=legal[0]).flatten())
                if s <= 1e-8:
                    m = np.zeros(7, dtype=np.float32); m[legal] = 1.0 / len(legal)
                col = int(np.random.choice(7, p=m))
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
        desc=f"{agent_a.name[:18]} vs {agent_b.name[:18]} (GPU)",
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

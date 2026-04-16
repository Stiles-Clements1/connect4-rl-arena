"""
game_engine.py — Connect-4 rules engine.

Board convention (matches all three group backends):
  - numpy array, shape (6, 7), dtype int8
  - +1  = red player (goes first on an empty board)
  - -1  = yellow player
  -  0  = empty cell
  - Row 0 is the TOP of the board; row 5 is the BOTTOM.
    A dropped piece falls to the highest-numbered empty row in its column.

This module is encoding-agnostic: it works entirely with the canonical
+1/−1/0 representation and never touches model inputs.
"""
import random as _random
import numpy as np

ROWS = 6
COLS = 7


def make_move(board: np.ndarray, col: int, player: int) -> np.ndarray:
    """
    Return a copy of `board` with `player` (+1 or -1) dropped into `col`.
    The piece falls to the lowest empty row (row 5 = bottom).
    Raises ValueError if the column is full.
    """
    board = board.copy()
    for row in range(ROWS - 1, -1, -1):   # scan from bottom upward
        if board[row, col] == 0:
            board[row, col] = player
            return board
    raise ValueError(f"Column {col} is full.")


def legal_moves(board: np.ndarray) -> list:
    """Return a list of column indices (0–6) whose top cell is empty."""
    return [c for c in range(COLS) if board[0, c] == 0]


def check_win(board: np.ndarray, player: int) -> bool:
    """Return True if `player` has four consecutive pieces anywhere on the board."""
    b = board
    # Horizontal
    for r in range(ROWS):
        for c in range(COLS - 3):
            if b[r, c] == b[r, c+1] == b[r, c+2] == b[r, c+3] == player:
                return True
    # Vertical
    for r in range(ROWS - 3):
        for c in range(COLS):
            if b[r, c] == b[r+1, c] == b[r+2, c] == b[r+3, c] == player:
                return True
    # Diagonal down-right
    for r in range(ROWS - 3):
        for c in range(COLS - 3):
            if b[r, c] == b[r+1, c+1] == b[r+2, c+2] == b[r+3, c+3] == player:
                return True
    # Diagonal down-left
    for r in range(ROWS - 3):
        for c in range(3, COLS):
            if b[r, c] == b[r+1, c-1] == b[r+2, c-2] == b[r+3, c-3] == player:
                return True
    return False


def is_terminal(board: np.ndarray):
    """
    Return (done, winner).
      done   : True if the game is over.
      winner : +1 or -1 if someone won, 0 for a draw, None if still playing.
    """
    if check_win(board, +1):
        return True, +1
    if check_win(board, -1):
        return True, -1
    if not legal_moves(board):
        return True, 0    # board full → draw
    return False, None


def winning_move(board: np.ndarray, player: int):
    """
    Return the first column that gives `player` an immediate four-in-a-row, or None.
    Used to make M2 a stronger opponent (per the assignment's adversarial training note).
    """
    for col in legal_moves(board):
        if check_win(make_move(board, col, player), player):
            return col
    return None


def blocking_move(board: np.ndarray, player: int):
    """
    Return the first column that blocks the opponent's immediate win, or None.
    Used to make M2 a stronger opponent (per the assignment's adversarial training note).
    """
    opponent = -player
    for col in legal_moves(board):
        if check_win(make_move(board, col, opponent), opponent):
            return col
    return None


def random_moves(board: np.ndarray, n: int, first_player: int = +1):
    """
    Apply up to `n` random legal moves starting with `first_player`, alternating.
    Stops early if the game ends before n moves are played.

    Returns:
        (new_board, next_player)  — next_player is None if the game ended
                                    during the warm-up phase.
    """
    board = board.copy()
    player = first_player
    for _ in range(n):
        moves = legal_moves(board)
        if not moves:
            break
        col = _random.choice(moves)
        board = make_move(board, col, player)
        done, _ = is_terminal(board)
        if done:
            # Signal to the caller that the game is already over
            return board, None
        player = -player
    return board, player

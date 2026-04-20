"""
Connect4 game logic
Board representation: 6x7 list of lists
+1 = human/player 1, -1 = bot/player 2, 0 = empty
"""
import numpy as np


def new_board():
    """Create empty 6x7 board"""
    return np.zeros((6, 7), dtype=np.float32).tolist()


def get_legal_moves(board):
    """Return list of columns (0-6) that aren't full"""
    legal = []
    for col in range(7):
        if board[0][col] == 0:  # Top row is empty
            legal.append(col)
    return legal


def apply_move(board, col, player):
    """
    Drop a piece in the given column for the player.
    Returns new board state (does not modify input).
    Raises ValueError if move is illegal.
    """
    # Convert to numpy for easier manipulation
    b = np.array(board, dtype=np.float32)

    # Check if column is full
    if b[0, col] != 0:
        raise ValueError(f"Column {col} is full")

    # Find lowest empty row
    for row in range(5, -1, -1):
        if b[row, col] == 0:
            b[row, col] = player
            break

    return b.tolist()


def check_win(board, player):
    """Check if player has won"""
    b = np.array(board)

    # Check horizontal
    for row in range(6):
        for col in range(4):
            if all(b[row, col+i] == player for i in range(4)):
                return True

    # Check vertical
    for row in range(3):
        for col in range(7):
            if all(b[row+i, col] == player for i in range(4)):
                return True

    # Check diagonal (down-right)
    for row in range(3):
        for col in range(4):
            if all(b[row+i, col+i] == player for i in range(4)):
                return True

    # Check diagonal (down-left)
    for row in range(3):
        for col in range(3, 7):
            if all(b[row+i, col-i] == player for i in range(4)):
                return True

    return False


def is_game_over(board):
    """
    Check if game is over.
    Returns (done, winner) where:
    - done is True if game is over
    - winner is +1, -1, 0 (draw), or None (not over)
    """
    # Check for wins
    if check_win(board, 1):
        return True, 1
    if check_win(board, -1):
        return True, -1

    # Check for draw (no legal moves)
    if len(get_legal_moves(board)) == 0:
        return True, 0

    return False, None

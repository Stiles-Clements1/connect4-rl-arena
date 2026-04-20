import argparse
import numpy as np
import tensorflow as tf
from pathlib import Path

ROWS, COLS = 6, 7


MODEL_PATH = Path("/anvilfolder/ubuntu/main/transformer_v2.keras")

print("USING MODEL_PATH:", MODEL_PATH, "exists:", MODEL_PATH.exists(), flush=True)
try:
    import anvil.server
    ANVIL_AVAILABLE = True
except Exception:
    ANVIL_AVAILABLE = False


def valid_moves(board: np.ndarray):
    return [c for c in range(COLS) if board[0, c] == 0]


def encode_board_optionB(board: np.ndarray):
    p0 = (board == +1).astype(np.int8)
    p1 = (board == -1).astype(np.int8)
    return np.stack([p0, p1], axis=-1)


def flip_perspective_optionB(board_2ch: np.ndarray):
    return board_2ch[..., ::-1].astype(np.int8)


class ModelServer:
    def __init__(self, model_path: Path):
        self.model = tf.keras.models.load_model(str(model_path))

    def predict_move(self, board, player=+1, encoding="B"):
        board = np.array(board, dtype=np.int8)

        if encoding == "A":
            board_in = board if player == +1 else (-board).astype(np.int8)
            x = board_in[np.newaxis, ..., np.newaxis].astype(np.float32)
        else:
            boardB = encode_board_optionB(board)
            board_in = boardB if player == +1 else flip_perspective_optionB(boardB)
            x = board_in[np.newaxis, ...].astype(np.float32)

        probs = self.model.predict(x, verbose=0)[0]
        moves = valid_moves(board)
        if not moves:
            return None

        masked = np.full_like(probs, -1e9)
        masked[moves] = probs[moves]
        return int(np.argmax(masked))


server_cache = {}


def get_server(model_path: Path):
    global server_cache
    key = str(model_path)

    if key not in server_cache:
        print(f"Loading model from {model_path}", flush=True)
        server_cache[key] = ModelServer(model_path=model_path)

    return server_cache[key]


def get_move(board, player=+1, model_path: Path = MODEL_PATH, encoding="B"):
    srv = get_server(model_path=model_path)
    return srv.predict_move(board, player=player, encoding=encoding)


if ANVIL_AVAILABLE:
    @anvil.server.callable
    def anvil_get_move(board, player=+1, model_path="transformer_v2.keras", encoding="B"):
        # If caller passed only 2 args like (board, "cnn_v2.keras"),
        # then "player" will actually be the model filename string.
        if isinstance(player, str) and model_path == "transformer_v2.keras":
            model_path, player = player, +1

        full_path = Path("/anvilfolder/ubuntu/main") / str(model_path)
        full_path = full_path.resolve()

        print("anvil_get_move called with:", {
            "player": player,
            "model_path_arg": model_path,
            "full_path": str(full_path),
            "exists": full_path.exists(),
            "encoding": encoding
        }, flush=True)

        return get_move(board, player=player, model_path=full_path, encoding=encoding)


def main():
    parser = argparse.ArgumentParser(description="Connect4 model backend")
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--model-path", default=str(MODEL_PATH))
    args = parser.parse_args()

    model_path = Path(args.model_path)

    if args.test:
        empty_board = np.zeros((ROWS, COLS), dtype=np.int8)
        move = get_move(empty_board, player=+1, model_path=model_path, encoding="B")
        print(f"CLI test move (empty board): {move}")
        return

    if ANVIL_AVAILABLE:
        anvil.server.connect("API KEY")
        anvil.server.wait_forever()
    else:
        print("Anvil not available. Use get_move locally.")


if __name__ == "__main__":
    main()

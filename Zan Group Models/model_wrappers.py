"""
Model wrapper functions for CNN and Transformer models
UPDATED VERSION - Uses new Transformer weights format with custom layers
"""
import tensorflow as tf
from tensorflow.keras import layers, Model
import numpy as np
import os


# ============================================
# CUSTOM LAYERS FOR TRANSFORMER
# ============================================

class PositionalEmbedding(layers.Layer):
    """Adds learned positional embeddings to input patches."""
    def __init__(self, num_patches, hidden_dim, **kwargs):
        super().__init__(**kwargs)
        self.num_patches = num_patches
        self.hidden_dim = hidden_dim

    def build(self, input_shape):
        self.embedding = self.add_weight(
            shape=(1, self.num_patches, self.hidden_dim),
            initializer="random_normal",
            trainable=True,
            name="pos_embedding"
        )
        super().build(input_shape)

    def call(self, x):
        return self.embedding

    def get_config(self):
        config = super().get_config()
        config.update({"num_patches": self.num_patches, "hidden_dim": self.hidden_dim})
        return config


class ClassToken(layers.Layer):
    """Prepends a learnable class token to the sequence."""
    def __init__(self, hidden_dim, **kwargs):
        super().__init__(**kwargs)
        self.hidden_dim = hidden_dim

    def build(self, input_shape):
        self.class_token = self.add_weight(
            shape=(1, 1, self.hidden_dim),
            initializer="random_normal",
            trainable=True,
            name="class_token"
        )
        super().build(input_shape)

    def call(self, x):
        batch_size = tf.shape(x)[0]
        class_token = tf.broadcast_to(self.class_token, [batch_size, 1, self.hidden_dim])
        return tf.concat([class_token, x], axis=1)

    def get_config(self):
        config = super().get_config()
        config.update({"hidden_dim": self.hidden_dim})
        return config


class TransformerBlock(layers.Layer):
    """A single transformer encoder block with pre-norm."""
    def __init__(self, hidden_dim, num_heads, mlp_dim, dropout_rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.mlp_dim = mlp_dim
        self.dropout_rate = dropout_rate

    def build(self, input_shape):
        self.ln1 = layers.LayerNormalization(epsilon=1e-6)
        self.mha = layers.MultiHeadAttention(
            num_heads=self.num_heads,
            key_dim=self.hidden_dim // self.num_heads,
            dropout=self.dropout_rate
        )
        self.ln2 = layers.LayerNormalization(epsilon=1e-6)
        self.mlp_dense1 = layers.Dense(self.mlp_dim, activation="gelu")
        self.mlp_dropout1 = layers.Dropout(self.dropout_rate)
        self.mlp_dense2 = layers.Dense(self.hidden_dim)
        self.mlp_dropout2 = layers.Dropout(self.dropout_rate)
        super().build(input_shape)

    def call(self, x, training=None):
        ln1_out = self.ln1(x)
        attn_out = self.mha(ln1_out, ln1_out, training=training)
        x = x + attn_out
        ln2_out = self.ln2(x)
        mlp_out = self.mlp_dense1(ln2_out)
        mlp_out = self.mlp_dropout1(mlp_out, training=training)
        mlp_out = self.mlp_dense2(mlp_out)
        mlp_out = self.mlp_dropout2(mlp_out, training=training)
        return x + mlp_out

    def get_config(self):
        config = super().get_config()
        config.update({
            "hidden_dim": self.hidden_dim,
            "num_heads": self.num_heads,
            "mlp_dim": self.mlp_dim,
            "dropout_rate": self.dropout_rate
        })
        return config


class LegalMask(tf.keras.layers.Layer):
    """Custom LegalMask layer for older model compatibility"""
    def __init__(self, **kwargs):
        super(LegalMask, self).__init__(**kwargs)

    def call(self, inputs):
        return inputs

    def get_config(self):
        config = super(LegalMask, self).get_config()
        return config


# ============================================
# TRANSFORMER MODEL BUILDER
# ============================================

def build_connect4_transformer(
    num_rows=6, num_cols=7, patch_features=2,
    hidden_dim=128, num_layers=6, num_heads=8,
    mlp_dim=256, dropout_rate=0.1
):
    """Build the Connect-4 Transformer architecture."""
    num_patches = num_rows * num_cols  # 42

    inp = layers.Input(shape=(num_patches, patch_features), name="board_input")
    x = layers.Dense(hidden_dim, name="patch_projection")(inp)

    pos_emb = PositionalEmbedding(num_patches, hidden_dim, name="pos_embedding")(inp)
    x = layers.Add()([x, pos_emb])
    x = layers.Dropout(dropout_rate)(x)
    x = ClassToken(hidden_dim, name="class_token_layer")(x)

    for i in range(num_layers):
        x = TransformerBlock(hidden_dim, num_heads, mlp_dim, dropout_rate,
                            name=f"transformer_block_{i}")(x)

    x = layers.LayerNormalization(epsilon=1e-6, name="final_ln")(x)

    # Policy head
    patch_tokens = layers.Lambda(lambda t: t[:, 1:, :], name="extract_patches")(x)
    patch_tokens = layers.Reshape((num_rows, num_cols, hidden_dim),
                                  name="reshape_patches")(patch_tokens)
    col_features = layers.Lambda(lambda t: tf.reduce_mean(t, axis=1),
                                 name="col_pooling")(patch_tokens)
    policy_hidden = layers.Dense(64, activation="relu", name="policy_dense1")(col_features)
    policy_logits = layers.Dense(1, name="policy_dense2")(policy_hidden)
    policy_logits = layers.Flatten(name="policy_flatten")(policy_logits)
    policy_output = layers.Softmax(name="policy")(policy_logits)

    # Value head
    class_token_out = layers.Lambda(lambda t: t[:, 0, :], name="extract_class")(x)
    value_hidden = layers.Dense(128, activation="relu", name="value_dense1")(class_token_out)
    value_hidden = layers.Dropout(dropout_rate)(value_hidden)
    value_hidden = layers.Dense(64, activation="relu", name="value_dense2")(value_hidden)
    value_output = layers.Dense(1, activation="tanh", name="value")(value_hidden)

    return Model(inputs=inp, outputs=[policy_output, value_output], name="Connect4_Transformer")


# ============================================
# GLOBAL MODEL INSTANCES
# ============================================

_cnn_model = None
_transformer_model = None


# ============================================
# MODEL LOADING FUNCTIONS
# ============================================

def load_cnn_model():
    """Load CNN model with proper handling"""
    global _cnn_model

    if _cnn_model is not None:
        return _cnn_model

    try:
        model_paths = [
            "models/cnn.h5",
            "/app/models/cnn.h5",
            "cnn.h5",
            "../models/cnn.h5"
        ]

        model_path = None
        for path in model_paths:
            if os.path.exists(path):
                model_path = path
                print(f"📁 Found CNN model at: {path}")
                break

        if model_path is None:
            print("❌ CNN model file not found in any expected location")
            print(f"   Searched: {model_paths}")
            return None

        print(f"📦 Loading CNN model from {model_path}...")
        _cnn_model = tf.keras.models.load_model(
            model_path,
            compile=False,
            custom_objects={'LegalMask': LegalMask}
        )

        print(f"✅ CNN model loaded successfully")
        print(f"   Input shape: {_cnn_model.input_shape}")

        if isinstance(_cnn_model.output, list):
            print(f"   Output shapes: {[o.shape for o in _cnn_model.output]}")
        else:
            print(f"   Output shape: {_cnn_model.output_shape}")

        return _cnn_model

    except Exception as e:
        print(f"❌ Failed to load CNN model: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


def load_transformer_model():
    """Load Transformer model by building architecture and loading weights"""
    global _transformer_model

    if _transformer_model is not None:
        return _transformer_model

    try:
        # Look for weights file (new format)
        weights_paths = [
            "models/transformer.weights.h5",
            "/app/models/transformer.weights.h5",
            "transformer.weights.h5",
            "../models/transformer.weights.h5"
        ]

        weights_path = None
        for path in weights_paths:
            if os.path.exists(path):
                weights_path = path
                print(f"📁 Found Transformer weights at: {path}")
                break

        if weights_path is None:
            print("❌ Transformer weights file not found")
            print(f"   Searched: {weights_paths}")
            return None

        # Build the model architecture first
        print(f"🔧 Building Transformer architecture...")
        _transformer_model = build_connect4_transformer(
            hidden_dim=128,
            num_layers=6,
            num_heads=8,
            mlp_dim=256,
            dropout_rate=0.1,
        )

        # Load the weights
        print(f"📦 Loading Transformer weights from {weights_path}...")
        _transformer_model.load_weights(weights_path)

        print(f"✅ Transformer model loaded successfully")
        print(f"   Input shape: {_transformer_model.input_shape}")
        print(f"   Output shapes: policy + value heads")

        return _transformer_model

    except Exception as e:
        print(f"❌ Failed to load Transformer model: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


# ============================================
# INPUT PREPARATION FUNCTIONS
# ============================================

def prepare_cnn_input(board):
    """
    Prepare board input for CNN model prediction.
    CNN expects (batch, 6, 7, 2) - one-hot encoded channels.

    Args:
        board: 6x7 numpy array with values 1, -1, 0

    Returns:
        Properly formatted input tensor (1, 6, 7, 2)
    """
    input_tensor = np.zeros((1, 6, 7, 2), dtype=np.float32)
    input_tensor[0, :, :, 0] = (board == 1).astype(np.float32)   # Current player pieces
    input_tensor[0, :, :, 1] = (board == -1).astype(np.float32)  # Opponent pieces
    return input_tensor


def prepare_transformer_input(board):
    """
    Prepare board input for Transformer model prediction.
    Transformer expects (batch, 42, 2) - flattened board with 2 channels.

    Args:
        board: 6x7 numpy array with values 1, -1, 0

    Returns:
        Properly formatted input tensor (1, 42, 2)
    """
    # Create 2-channel representation: channel 0 = current player, channel 1 = opponent
    input_tensor = np.zeros((1, 42, 2), dtype=np.float32)
    board_flat = board.flatten()
    input_tensor[0, :, 0] = (board_flat == 1).astype(np.float32)   # Current player pieces
    input_tensor[0, :, 1] = (board_flat == -1).astype(np.float32)  # Opponent pieces
    return input_tensor


def get_valid_moves(board):
    """Get list of valid moves (columns that aren't full)"""
    valid_moves = []
    for col in range(7):
        if board[0, col] == 0:
            valid_moves.append(col)
    return valid_moves


# ============================================
# PREDICTION FUNCTIONS
# ============================================

def extract_policy_from_output(outputs):
    """
    Extract policy (move probabilities) from model output.
    Handles various output formats.
    """
    if isinstance(outputs, dict):
        if 'policy' in outputs:
            policy = outputs['policy']
        elif 'output_policy' in outputs:
            policy = outputs['output_policy']
        else:
            policy = list(outputs.values())[0]
    elif isinstance(outputs, (list, tuple)):
        # First output is typically policy
        policy = outputs[0]
    else:
        policy = outputs

    policy = np.array(policy).flatten()

    if len(policy) >= 7:
        policy = policy[:7]
    else:
        policy = np.concatenate([policy, np.zeros(7 - len(policy))])

    return policy


def predict_cnn_move(model, board):
    """Predict move using CNN model"""
    try:
        input_tensor = prepare_cnn_input(board)
        outputs = model.predict(input_tensor, verbose=0)
        move_probs = extract_policy_from_output(outputs)
        valid_moves = get_valid_moves(board)

        if not valid_moves:
            print("⚠️ No valid moves available")
            return 3

        # Mask invalid moves
        masked_probs = np.full(7, -np.inf)
        for move in valid_moves:
            masked_probs[move] = move_probs[move]

        predicted_move = int(np.argmax(masked_probs))

        if predicted_move not in valid_moves:
            print(f"⚠️ Predicted move {predicted_move} not valid, choosing random")
            predicted_move = int(np.random.choice(valid_moves))

        return predicted_move

    except Exception as e:
        print(f"❌ CNN prediction error: {str(e)}")
        valid_moves = get_valid_moves(board)
        return int(np.random.choice(valid_moves)) if valid_moves else 3


def predict_transformer_move(model, board):
    """Predict move using Transformer model"""
    try:
        input_tensor = prepare_transformer_input(board)
        policy, value = model.predict(input_tensor, verbose=0)

        move_probs = policy[0]  # Shape: (7,)
        valid_moves = get_valid_moves(board)

        if not valid_moves:
            print("⚠️ No valid moves available")
            return 3

        # Mask invalid moves
        masked_probs = np.full(7, -np.inf)
        for move in valid_moves:
            masked_probs[move] = move_probs[move]

        predicted_move = int(np.argmax(masked_probs))

        if predicted_move not in valid_moves:
            print(f"⚠️ Predicted move {predicted_move} not valid, choosing random")
            predicted_move = int(np.random.choice(valid_moves))

        return predicted_move

    except Exception as e:
        print(f"❌ Transformer prediction error: {str(e)}")
        import traceback
        traceback.print_exc()
        valid_moves = get_valid_moves(board)
        return int(np.random.choice(valid_moves)) if valid_moves else 3


# ============================================
# PUBLIC API FUNCTIONS
# ============================================

def get_cnn_move(board):
    """Get move from CNN model"""
    global _cnn_model

    if _cnn_model is None:
        _cnn_model = load_cnn_model()

    if _cnn_model is None:
        print("❌ CNN model not available, using random")
        valid_moves = get_valid_moves(board)
        return int(np.random.choice(valid_moves)) if valid_moves else 3

    return predict_cnn_move(_cnn_model, board)


def get_transformer_move(board):
    """Get move from Transformer model"""
    global _transformer_model

    if _transformer_model is None:
        _transformer_model = load_transformer_model()

    if _transformer_model is None:
        print("❌ Transformer model not available, using random")
        valid_moves = get_valid_moves(board)
        return int(np.random.choice(valid_moves)) if valid_moves else 3

    return predict_transformer_move(_transformer_model, board)


# ============================================
# TEST CODE
# ============================================

if __name__ == "__main__":
    print("=" * 50)
    print("🧪 Testing Model Wrappers")
    print("=" * 50)

    # Test board
    test_board = np.zeros((6, 7))
    test_board[5, 3] = 1   # Center piece
    test_board[5, 2] = -1  # Opponent piece

    print(f"📋 Test board:\n{test_board}")
    print()

    # Test CNN
    print("🤖 Testing CNN Model:")
    cnn_move = get_cnn_move(test_board)
    print(f"   Recommended move: {cnn_move}")
    print()

    # Test Transformer
    print("🤖 Testing Transformer Model:")
    transformer_move = get_transformer_move(test_board)
    print(f"   Recommended move: {transformer_move}")
    print()

    print("=" * 50)
    print("🎉 Model Wrapper Testing Complete!")
    print("=" * 50)

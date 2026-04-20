"""
AI inference for Connect4 bots - WORKING VERSION
Custom layers with proper build() methods to avoid warnings
"""
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


# Custom layers - properly implemented to avoid build warnings
class AddPositionEmb(layers.Layer):
    def __init__(self, num_patches, projection_dim, **kwargs):
        super().__init__(**kwargs)
        self.num_patches = num_patches
        self.projection_dim = projection_dim

    def build(self, input_shape):
        self.pos_emb = layers.Embedding(
            input_dim=self.num_patches,
            output_dim=self.projection_dim
        )
        self.pos_emb.build((None, self.num_patches))
        super().build(input_shape)

    def call(self, x):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        return x + self.pos_emb(positions)

    def get_config(self):
        config = super().get_config()
        config.update({
            "num_patches": self.num_patches,
            "projection_dim": self.projection_dim,
        })
        return config


class ClassToken(layers.Layer):
    def __init__(self, projection_dim, **kwargs):
        super().__init__(**kwargs)
        self.projection_dim = projection_dim

    def build(self, input_shape):
        self.cls_token = self.add_weight(
            shape=(1, 1, self.projection_dim),
            initializer='random_normal',
            trainable=True,
            name='cls_token'
        )
        super().build(input_shape)

    def call(self, x):
        batch_size = tf.shape(x)[0]
        cls_tokens = tf.tile(self.cls_token, [batch_size, 1, 1])
        return tf.concat([cls_tokens, x], axis=1)

    def get_config(self):
        config = super().get_config()
        config.update({
            "projection_dim": self.projection_dim,
        })
        return config


# Global model storage
_models = {}


def load_models():
    """
    Load both models once at startup.
    Models are stored in _models dict.
    """
    models_dir = os.path.join(os.path.dirname(__file__), '..', 'models')

    cnn_path = os.path.join(models_dir, 'best_cnn_model.keras')
    transformer_keras = os.path.join(models_dir, 'best_transformer_model.keras')
    transformer_saved = os.path.join(models_dir, 'best_transformer_model')

    custom_objs = {
        'AddPositionEmb': AddPositionEmb,
        'ClassToken': ClassToken
    }

    print(f"Loading CNN model from {cnn_path}...")
    _models['cnn'] = keras.models.load_model(cnn_path, compile=False)
    print(f"  CNN input shape: {_models['cnn'].input_shape}")
    print(f"  CNN output shape: {_models['cnn'].output_shape}")

    print(f"Loading Transformer model...")
    _models['transformer'] = None
    # Try SavedModel format first (better cross-version compatibility)
    if os.path.isdir(transformer_saved) and os.path.exists(
        os.path.join(transformer_saved, 'saved_model.pb')
    ):
        try:
            _models['transformer'] = keras.models.load_model(
                transformer_saved,
                custom_objects=custom_objs,
                compile=False,
            )
            print(f"  Loaded from SavedModel: {transformer_saved}")
        except Exception as e:
            print(f"  SavedModel load failed: {e}")
    # Fall back to .keras
    if _models['transformer'] is None and os.path.exists(transformer_keras):
        try:
            _models['transformer'] = keras.models.load_model(
                transformer_keras,
                custom_objects=custom_objs,
                compile=False,
            )
            print(f"  Loaded from .keras: {transformer_keras}")
        except Exception as e:
            print(f"  .keras load failed: {e}")

    if _models['transformer'] is not None:
        print(f"  Transformer input shape: {_models['transformer'].input_shape}")
        print(f"  Transformer output shape: {_models['transformer'].output_shape}")
        print("✓ Both models loaded successfully!")
    else:
        print(f"  WARNING: Could not load Transformer model. Only CNN available.")
        print("✓ CNN model loaded successfully (Transformer unavailable)")


def encode_board(board):
    """
    Convert board to model input format.
    Board is 6x7 list of lists, needs to be (1, 6, 7, 1) numpy array.
    """
    b = np.array(board, dtype=np.float32)
    return b.reshape(1, 6, 7, 1)


def get_bot_move(board, legal_moves, bot_type='cnn', player=-1):
    """
    Get AI's best legal move using the specified model.
    """
    if bot_type not in _models:
        raise ValueError(f"Unknown bot type: {bot_type}. Must be 'cnn' or 'transformer'")

    model = _models[bot_type]

    if model is None:
        raise ValueError(
            f"{bot_type} model is not available. "
            "It could not be loaded. Use 'cnn' instead."
        )

    board_array = np.array(board, dtype=np.float32)
    board_input = board_array * -1 if player == -1 else board_array.copy()

    encoded = encode_board(board_input)
    predictions = model.predict(encoded, verbose=0)[0]

    legal_predictions = [(col, float(predictions[col])) for col in legal_moves]
    best_move = max(legal_predictions, key=lambda x: x[1])[0]
    return int(best_move)

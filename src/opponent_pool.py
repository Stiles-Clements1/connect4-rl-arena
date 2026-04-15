"""
opponent_pool.py — manages the growing set of M2 opponents.

Rules (from the assignment):
  - Original MCTS-mimicking networks are NEVER removed from the pool.
  - Every POOL_ADD_INTERVAL training groups, a frozen snapshot of the current
    M1 weights is cloned and added to the pool (unless the pool is at POOL_CAP).
  - The pool never exceeds POOL_CAP entries (originals always count against the cap,
    so add M1 copies only while there is room).
"""
import numpy as np
import tensorflow as tf

from .model_loader import ModelWrapper
from . import config as _cfg


class OpponentPool:
    """
    Maintains a list of ModelWrapper opponents for M2 sampling.

    The first `_num_original` entries are the initial MCTS networks and are
    never removed, regardless of POOL_CAP.
    """

    def __init__(self, initial_wrappers: list):
        """
        Seed the pool with the initial set of M2 wrappers.
        `initial_wrappers` should NOT include the M1 wrapper being trained —
        M1 copies are added later via maybe_add_m1_copy().
        """
        self._pool = list(initial_wrappers)
        # Bookmark: entries below this index are originals and never removed
        self._num_original = len(initial_wrappers)

    def sample(self) -> ModelWrapper:
        """Return a uniformly random ModelWrapper from the pool."""
        return self._pool[np.random.randint(len(self._pool))]

    def maybe_add_m1_copy(self, m1_wrapper: ModelWrapper, group: int) -> bool:
        """
        If `group` is a multiple of POOL_ADD_INTERVAL and the pool has room,
        clone M1's current weights into a new frozen wrapper and append it.

        Returns True if a copy was added, False otherwise.
        """
        if group % _cfg.POOL_ADD_INTERVAL != 0:
            return False
        if len(self._pool) >= _cfg.POOL_CAP:
            return False

        # Clone the model architecture and copy current trained weights
        copy_model = tf.keras.models.clone_model(m1_wrapper.model)
        copy_model.set_weights(m1_wrapper.model.get_weights())

        copy_wrapper = ModelWrapper(
            model=copy_model,
            encoding=m1_wrapper.encoding,
            name=f"M1_snapshot_g{group}",
        )
        self._pool.append(copy_wrapper)
        return True

    def __len__(self) -> int:
        return len(self._pool)

    def __repr__(self) -> str:
        names = [w.name for w in self._pool]
        return f"OpponentPool(size={len(self._pool)}, opponents={names})"

"""
neurofm/losses.py

Loss functions and metrics registry. These dicts are passed to
get_custom_objects() so TensorFlow can deserialise the model from
a saved .h5 file that was compiled with these losses during training.
"""
from __future__ import annotations

import tensorflow as tf


def huber_loss(delta=1.0):
    """Returns a Huber loss function with the specified delta."""
    def loss(y_true, y_pred):
        return tf.keras.losses.huber(y_true, y_pred, delta)
    return loss

losses_dict = {
    "categorical_crossentropy": tf.keras.losses.categorical_crossentropy,
    "mean_squared_error": tf.keras.losses.mean_squared_error,
    "mean_absolute_error": tf.keras.losses.mean_absolute_error,
    "log_cosh": tf.keras.losses.log_cosh,
    "huber_5": huber_loss(delta=5),
    "huber_1": huber_loss(delta=1),
    "huber_0_4": huber_loss(delta=0.4),
    "huber_0_1": huber_loss(delta=0.1),
}

metrics_dict = {
    "categorical_crossentropy": tf.keras.metrics.categorical_crossentropy,
    "mean_squared_error": tf.keras.metrics.mean_squared_error,
    "mean_absolute_error": tf.keras.metrics.mean_absolute_error,
    "log_cosh": tf.keras.metrics.log_cosh,
    "accuracy": tf.keras.metrics.categorical_accuracy,
}
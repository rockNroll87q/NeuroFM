"""
neurofm/model.py

Model architecture definition and weight loading for inference.

Training-specific functionality (finetuning, layer freezing, competing
architectures, resume training) has been intentionally stripped. This
module contains only what is needed to construct the model and load
pretrained weights.
"""
from __future__ import annotations

from typing import Literal

import tensorflow as tf
from loguru import logger
from pydantic import BaseModel
from tensorflow.keras import Model
from tensorflow.keras.layers import (
    Activation,
    BatchNormalization,
    Dense,
    Dropout,
    Flatten,
    GlobalAveragePooling3D,
)

from .layers import BottleNeck, Plain, Residual, add_conv_layer
from .losses import losses_dict, metrics_dict

# Internal order matches the trained model output heads exactly.
# Do not reorder — this must match predicted_variable in VARIANT_CONFIGS.
_BRAIN_HEALTH_INTERNAL = ["PatientAge", "PatientSex", "ventricular_volume", "brain_volume"]
 
# # User-facing column names in all output CSVs and .npy files.
BRAIN_HEALTH_KEYS = ["brain_age", "sex", "ventricle_volume", "brain_volume"]
 

# ---------------------------------------------------------------------------
# Network configuration
# ---------------------------------------------------------------------------

class NetworkConfig(BaseModel):
    """
    Configuration for a NeuroFM model variant.
    All hyperparameters required to reconstruct the architecture exactly
    as trained — do not change these for a given set of pretrained weights.

    Defaults match the training NetConfig from config.py exactly.
    """

    # Input
    shape: tuple[int, int, int] = (182, 218, 182)  # fixed MNI152 1mm space

    # Encoder
    conv_block: Literal["Plain", "BottleNeck", "Residual"] = "BottleNeck"
    num_conv_layers: int = 5
    num_initial_filter: int = 32
    conv_repetition: int = 2
    bottleneck_factor: int = 4                  # divide_factor in BottleNeck
    identity_layers: bool = False
    n_identity_layers: int = 3
    identity_layer_start: int = 3
    activation: str = "elu"
    bn: str = "BN"                              # "BN", "GN", or None
    kernel_initializer: str = "he_normal"
    kernel_regularizer: float = 1e-4
    dropout_rate: float = 0.05
    downsampling: str = "conv"                  # "conv" or "pooling"
    stride: int = 2
    use_se: bool = False
    se_ratio: float = 1.0

    # Pooling / bridge to neck
    final_stage: Literal["dense", "avgPool"] = "avgPool"
    num_dense_layers: int = 3                   # only used if final_stage == 'dense'

    # Neck
    num_neck_layers: int = 0
    neck_layer_size: int = 512

    # Output head
    num_classes: list[int] = [1, 2, 1, 1]       # one per predicted variable
    predicted_variable: list[str] = BRAIN_HEALTH_KEYS
    dense_predictors: bool = False
    num_dense_predictor_layers: int = 1
    dense_predictor_size: int = 128


# ---------------------------------------------------------------------------
# Per-variant configs
# ---------------------------------------------------------------------------
# These must exactly match the hyperparameters used during pretraining.
# Changing any value will produce incorrect outputs with the released weights.

VARIANT_CONFIGS: dict[str, NetworkConfig] = {
    "neurofm-s": NetworkConfig(
        num_conv_layers=5,
        num_initial_filter=32,
        num_neck_layers=0,
        dropout_rate=0.1,
    ),
    "neurofm-m": NetworkConfig(
        num_conv_layers=7,
        num_initial_filter=32,
        identity_layers=True,
        n_identity_layers=5,
        identity_layer_start=2,
        num_dense_layers=1,
        dense_predictors=True,
        neck_layer_size=256,
        num_neck_layers=2,
        dropout_rate=0.25,
    ),
    "neurofm-l": NetworkConfig(
        num_conv_layers=8,
        num_initial_filter=32,
        identity_layers=True,
        n_identity_layers=5,
        identity_layer_start=2,
        num_dense_layers=1,
        dense_predictors=True,
        dense_predictor_size=256,
        neck_layer_size=512,
        num_neck_layers=2,
        dropout_rate=0.25,
        final_stage="dense",
    ),
}

# ---------------------------------------------------------------------------
# Custom objects — required for loading saved .h5 models
# ---------------------------------------------------------------------------

def get_custom_objects() -> dict:
    """
    Returns the custom Keras objects dict needed to deserialise the model
    from a saved .h5 or SavedModel file.
    """
    custom_objects = {
        "Plain": Plain,
        "BottleNeck": BottleNeck,
        "Residual": Residual,
    }
    custom_objects.update(losses_dict)
    custom_objects.update(metrics_dict)
    return custom_objects


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_neurofm(weights_path: str, variant: str) -> Model:
    """
    Load a pretrained NeuroFM model from an .h5 weights file.

    Builds the architecture from the variant config, then loads
    the pretrained weights by name. This is the preferred approach
    over tf.keras.models.load_model() as it avoids serialisation
    issues with custom layers in TF 2.13.

    Parameters
    ----------
    weights_path : str
        Path to the .h5 weights file.
    variant : str
        One of the keys in VARIANT_CONFIGS.

    Returns
    -------
    tf.keras.Model
        Model with pretrained weights loaded, in inference mode.
    """
    if variant not in VARIANT_CONFIGS:
        raise ValueError(
            f"Unknown variant '{variant}'. Valid options: {list(VARIANT_CONFIGS.keys())}"
        )

    config = VARIANT_CONFIGS[variant]
    logger.info(f"Building {variant} architecture...")
    model = build_model(config)

    logger.info(f"Loading weights from {weights_path}...")
    model.load_weights(weights_path, by_name=True)

    logger.info(f"Model ready — {_count_params(model):,} parameters.")
    return model


# ---------------------------------------------------------------------------
# Architecture builders
# ---------------------------------------------------------------------------

_CONV_BLOCKS = {
    "Plain": Plain,
    "BottleNeck": BottleNeck,
    "Residual": Residual,
}


def build_model(config: NetworkConfig) -> Model:
    """
    Build the NeuroFM 3D inference model from a NetworkConfig.

    Parameters
    ----------
    config : NetworkConfig
        Architecture hyperparameters. Use one of the VARIANT_CONFIGS dicts
        for a released variant — do not change parameters for pretrained weights.

    Returns
    -------
    tf.keras.Model
    """
    input_shape = (*config.shape, 1)

    with tf.name_scope("Input"):
        input_layer = tf.keras.layers.Input(input_shape)

    x = input_layer
    x = _build_encoder_layers(config, x)
    x = _build_dense_pooling_layers(config, x)
    x = _build_neck_layers(config, x)
    output_layer = _build_output_layers(config, x)

    return Model(input_layer, output_layer)

def _build_encoder_layers(config: NetworkConfig, x):
    with tf.name_scope("Encoder"):
        for f in range(config.num_conv_layers):
            filter_mult = (2 ** f) if config.conv_block == "Residual" else (f + 1)
            n_identity = (
                config.n_identity_layers
                if config.identity_layers
                else 1
            )
            if f < config.identity_layer_start:
                n_identity = 1

            for r in range(n_identity):
                x = add_conv_layer(
                    config.conv_block,
                    x,
                    filter_num=config.num_initial_filter * filter_mult,
                    activation=config.activation,
                    bn=config.bn,
                    groups=8,
                    kernel_initializer=config.kernel_initializer,
                    kernel_regularizer=config.kernel_regularizer,
                    n_conv_row=config.conv_repetition,
                    divide_factor=config.bottleneck_factor,
                    dropout_rate=config.dropout_rate,
                    use_se=config.use_se,
                    se_ratio=config.se_ratio,
                    downsampling=config.downsampling,
                    stride=1 if r < n_identity - 1 else config.stride,
                    name=f"enc_conv_{f + 1}_{r + 1}",
                )
    return x


def _build_dense_pooling_layers(config: NetworkConfig, x):
    if config.final_stage == "dense":
        x = Flatten()(x)
        with tf.name_scope("FC"):
            for f in range(config.num_dense_layers):
                x = Dense(1024, activation=config.activation, name=f"enc_dense_{f}")(x)
                x = Dropout(rate=config.dropout_rate, name=f"dropout_dense_{f}")(x)

    elif config.final_stage == "avgPool":
        with tf.name_scope("GlobalAveragePooling"):
            x = GlobalAveragePooling3D(
                name=f"GlobalAveragePooling_{config.num_conv_layers - 1}"
            )(x)
    else:
        raise ValueError(
            f"Unrecognised final_stage '{config.final_stage}'. "
            "Expected 'dense' or 'avgPool'."
        )
    return x


def _build_neck_layers(config: NetworkConfig, x):
    with tf.name_scope("Neck"):
        for n in range(config.num_neck_layers):
            x = Dense(
                config.neck_layer_size,
                activation=config.activation,
                name=f"enc_neck_{n}",
            )(x)
            x = BatchNormalization(name=f"bn_neck_{n}")(x)
            x = Dropout(config.dropout_rate, name=f"dropout_neck_{n}")(x)
    return x


def _build_output_layers(config: NetworkConfig, x):
    output_layers = []
    # Latent feature extraction point — named layer referenced in inference.py
    x_multihead = tf.keras.layers.Layer(name="multihead_output")(x)

    with tf.name_scope("Output"):
        for idx, n_classes in enumerate(config.num_classes):
            x = x_multihead
            var_name = config.predicted_variable[idx]

            if config.dense_predictors:
                for d in range(config.num_dense_predictor_layers):
                    x = Dense(
                        config.dense_predictor_size,
                        activation=config.activation,
                        name=f"enc_dense_pred_{var_name}_{d}",
                    )(x)
                    x = BatchNormalization(
                        name=f"bn_dense_pred_{var_name}_{d}"
                    )(x)
                    x = Dropout(0.5, name=f"dropout_dense_pred_{var_name}_{d}")(x)

            last_activation = "linear" if n_classes == 1 else "softmax"
            x = Dense(n_classes)(x)
            # Force float32 output — avoids errors with mixed precision training
            x = Activation(last_activation, dtype="float32", name=var_name)(x)
            output_layers.append(x)

    return output_layers


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _count_params(model: Model) -> int:
    return int(sum(tf.size(w).numpy() for w in model.trainable_weights))

def get_mid_layer(model, layer_name):
    """
    Returns a new Keras model that has the outputs as the specified intermediate layer.
    This can be useful for extracting features or using the output of a specific 
    layer. Pass the returned object of this function to `extract_latent_features`, 
    for example.

    Parameters:
        - model (Keras model): The original model from which to extract
            the intermediate layer.
        - layer_name (str): The name of the intermediate layer to 
            extract.

    Returns:
        - intermediate_layer_model (Keras model): A new Keras model that 
            only goes up to the specified intermediate layer.
    """
    intermediate_layer_model = Model(inputs=model.input, outputs=model.get_layer(layer_name).output)
    return intermediate_layer_model

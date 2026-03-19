#!/usr/bin/env python3
"""
=============================================================================
Austin Dibble
University of Glasgow
2026

neurofm/layers.py

Custom Keras layers used in the NeuroFM architecture.

SSFAdaLayer and deconstructed layer variants (fn_BottleNeck, fn_PooledBottleNeck)
have been intentionally omitted — they are training/finetuning-only constructs
not required for inference.
=============================================================================
"""
from __future__ import annotations

import tensorflow as tf


class Plain(tf.keras.layers.Layer):
    """
    VGG-like encoder block for 3D tensors.

    Structure:
        Input -> [Conv > BN > Activation] x n_conv_row -> Downsample -> Dropout -> Activation
    """

    def __init__(
        self,
        filter_num: int,
        dropout_rate: float = 0.1,
        stride: int = 2,
        activation: str = "relu",
        bn: bool = True,
        groups: int = 8,
        kernel_initializer: str = "he_normal",
        kernel_regularizer: float = 1e-4,
        n_conv_row: int = 1,
        downsampling: str = "conv",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.activation = activation
        self.filter_num = filter_num
        self.dropout_rate = dropout_rate
        self.stride = stride
        self.bn = bn
        self.n_conv_row = n_conv_row
        self._downsampling_type = downsampling

        kernel_reg = tf.keras.regularizers.l2(kernel_regularizer)

        self.convs = tf.keras.Sequential()
        for _i in range(n_conv_row):
            self.convs.add(
                tf.keras.layers.Conv3D(
                    filters=filter_num,
                    kernel_size=(3, 3, 3),
                    strides=1,
                    padding="same",
                    kernel_initializer=kernel_initializer,
                    kernel_regularizer=kernel_reg,
                )
            )
            if bn == "BN":
                self.convs.add(tf.keras.layers.BatchNormalization(fused=False))
            elif bn == "GN":
                self.convs.add(
                    tf.keras.layers.GroupNormalization(groups=min(groups, filter_num))
                )
            self.convs.add(tf.keras.layers.Activation(activation))

        if downsampling == "conv":
            self.downsampling = tf.keras.layers.Conv3D(
                filters=filter_num * 2,
                kernel_size=(1, 1, 1),
                strides=stride,
                padding="same",
                kernel_initializer=kernel_initializer,
                kernel_regularizer=kernel_reg,
            )
        elif downsampling == "pooling":
            self.downsampling = tf.keras.layers.MaxPool3D(pool_size=stride)
        else:
            raise ValueError(
                f"Unrecognised downsampling strategy '{downsampling}'. "
                "Expected 'conv' or 'pooling'."
            )

        self.dropout = tf.keras.layers.Dropout(rate=dropout_rate)

    def __call__(self, inputs, training=None, **kwargs):
        """Run forward pass."""
        x = self.convs(inputs, training=training)
        if training:
            x = self.dropout(x)
        x = self.downsampling(x)
        x = getattr(tf.nn, self.activation)(x)
        return x

    def get_config(self):
        """Get layer config"""
        config = super().get_config()
        config.update(
            {
                "filter_num": self.filter_num,
                "dropout_rate": self.dropout_rate,
                "stride": self.stride,
                "activation": self.activation,
                "bn": self.bn,
                "n_conv_row": self.n_conv_row,
                "downsampling": self._downsampling_type,
            }
        )
        return config

    @classmethod
    def from_config(cls, config, custom_objects=None):
        """From given config"""
        return cls(**config)


class Residual(tf.keras.layers.Layer):
    """
    Residual encoder block for 3D tensors.

    Structure:
        Input -|-> [Conv > BN > Activation] x n_conv_row -> Add -> Downsample -> Dropout -> Activation
               \________________________________________________/
    """

    def __init__(
        self,
        filter_num: int,
        dropout_rate: float = 0.1,
        stride: int = 2,
        activation: str = "relu",
        bn: bool = True,
        groups: int = 8,
        kernel_initializer: str = "he_normal",
        kernel_regularizer: float = 1e-4,
        n_conv_row: int = 1,
        m_res_blocks: int = 1,
        downsampling: str = "conv",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.activation = activation
        self.filter_num = filter_num
        self.dropout_rate = dropout_rate
        self.stride = stride
        self.bn = bn
        self.n_conv_row = n_conv_row
        self.m_res_blocks = m_res_blocks
        self._downsampling_type = downsampling

        kernel_reg = tf.keras.regularizers.l2(kernel_regularizer)

        self.convs = tf.keras.Sequential()
        for i in range(n_conv_row):
            self.convs.add(
                tf.keras.layers.Conv3D(
                    filters=filter_num,
                    kernel_size=(3, 3, 3),
                    strides=1,
                    padding="same",
                    kernel_initializer=kernel_initializer,
                    kernel_regularizer=kernel_reg,
                )
            )
            if bn == "BN":
                self.convs.add(tf.keras.layers.BatchNormalization(fused=False))
            elif bn == "GN":
                self.convs.add(
                    tf.keras.layers.GroupNormalization(groups=min(groups, filter_num))
                )
            if i != (n_conv_row - 1):
                self.convs.add(tf.keras.layers.Activation(activation))

        if downsampling == "conv":
            self.downsampling = tf.keras.layers.Conv3D(
                filters=filter_num * 2,
                kernel_size=(1, 1, 1),
                strides=stride,
                padding="same",
                kernel_initializer=kernel_initializer,
                kernel_regularizer=kernel_reg,
            )
        elif downsampling == "pooling":
            self.downsampling = tf.keras.layers.MaxPool3D(pool_size=stride)
        else:
            raise ValueError(
                f"Unrecognised downsampling strategy '{downsampling}'. "
                "Expected 'conv' or 'pooling'."
            )

        self.dropout = tf.keras.layers.Dropout(rate=dropout_rate)

    def call(self, inputs, training=None, **kwargs):
        """Run forward pass."""
        x = self.convs(inputs)
        x = tf.keras.layers.add([inputs, x])
        x = self.downsampling(x)
        if training:
            x = self.dropout(x)
        x = getattr(tf.nn, self.activation)(x)
        return x

    def get_config(self):
        """Get layer config"""
        config = super().get_config()
        config.update(
            {
                "filter_num": self.filter_num,
                "dropout_rate": self.dropout_rate,
                "stride": self.stride,
                "activation": self.activation,
                "bn": self.bn,
                "n_conv_row": self.n_conv_row,
                "downsampling": self._downsampling_type,
            }
        )
        return config

    @classmethod
    def from_config(cls, config, custom_objects=None):
        """Create from a given config"""
        return cls(**config)


class BottleNeck(tf.keras.layers.Layer):
    """
    ResNet-like bottleneck encoder block for 3D tensors.

    Structure:
        Input -|-> Conv > BN > Act > Conv > BN > Act > Conv > BN > [SE] -> Dropout -> Add -> Activation
               \-> Conv > BN >__________________________________________________/
    """

    def __init__(
        self,
        filter_num: int = 32,
        dropout_rate: float = 0.1,
        stride: int = 2,
        activation: str = "relu",
        bn: bool = "bn",
        groups: int = 8,
        kernel_initializer: str = "he_normal",
        kernel_regularizer: float = 1e-4,
        n_conv_row: int = 1,
        divide_factor: int = 4,
        downsampling: str = "conv",
        use_se: bool = False,
        se_ratio: float = 1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.activation = activation
        self.filter_num = filter_num
        self.dropout_rate = dropout_rate
        self.stride = stride
        self.bn = bn
        self.groups = groups
        self.kernel_initializer = kernel_initializer
        self.kernel_regularizer = kernel_regularizer
        self.divide_factor = divide_factor
        self._downsampling_type = downsampling
        self.use_se = use_se
        self.se_ratio = se_ratio

        kernel_reg = tf.keras.regularizers.l2(kernel_regularizer)

        self.conv1 = tf.keras.layers.Conv3D(
            filters=filter_num,
            kernel_size=(1, 1, 1),
            strides=stride,
            padding="same",
            kernel_initializer=kernel_initializer,
            kernel_regularizer=kernel_reg,
        )
        self.conv2 = tf.keras.layers.Conv3D(
            filters=int(filter_num / divide_factor),
            kernel_size=(3, 3, 3),
            strides=1,
            padding="same",
            kernel_initializer=kernel_initializer,
            kernel_regularizer=kernel_reg,
        )
        self.conv3 = tf.keras.layers.Conv3D(
            filters=filter_num,
            kernel_size=(1, 1, 1),
            strides=1,
            padding="same",
            kernel_initializer=kernel_initializer,
            kernel_regularizer=kernel_reg,
        )

        if use_se:
            self.se = SqueezeExcitation(
                in_filters=filter_num,
                out_filters=filter_num,
                se_ratio=se_ratio,
                divisible_by=8,
                use_3d_input=True,
                kernel_initializer=kernel_initializer,
                activation=activation,
                gating_activation="sigmoid",
            )

        if bn == "BN":
            self.bn1 = tf.keras.layers.BatchNormalization(fused=False)
            self.bn2 = tf.keras.layers.BatchNormalization(fused=False)
            self.bn3 = tf.keras.layers.BatchNormalization(fused=False)
        elif bn == "GN":
            self.bn1 = tf.keras.layers.GroupNormalization(groups=min(groups, filter_num))
            self.bn2 = tf.keras.layers.GroupNormalization(groups=min(groups, filter_num))
            self.bn3 = tf.keras.layers.GroupNormalization(groups=min(groups, filter_num))

        self.dropout = tf.keras.layers.Dropout(rate=dropout_rate)

        self.downsample = tf.keras.Sequential()
        if downsampling == "conv":
            self.downsample.add(
                tf.keras.layers.Conv3D(
                    filters=filter_num,
                    kernel_size=(1, 1, 1),
                    strides=stride,
                    padding="same",
                    kernel_initializer=kernel_initializer,
                    kernel_regularizer=kernel_reg,
                )
            )
        elif downsampling == "pooling":
            self.downsample.add(tf.keras.layers.MaxPool3D(pool_size=stride))
        else:
            raise ValueError(
                f"Unrecognised downsampling strategy '{downsampling}'. "
                "Expected 'conv' or 'pooling'."
            )

        if bn == "BN":
            self.downsample.add(tf.keras.layers.BatchNormalization(fused=False))
        elif bn == "GN":
            self.downsample.add(
                tf.keras.layers.GroupNormalization(groups=min(groups, filter_num))
            )

    def call(self, inputs, training=None, **kwargs):
        """Run forward pass."""

        residual = self.downsample(inputs)

        x = self.conv1(inputs)
        if self.bn in ["BN", "GN"]:
            x = self.bn1(x, training=training)
        x = getattr(tf.nn, self.activation)(x)

        x = self.conv2(x)
        if self.bn in ["BN", "GN"]:
            x = self.bn2(x, training=training)
        x = getattr(tf.nn, self.activation)(x)

        x = self.conv3(x)
        if self.bn in ["BN", "GN"]:
            x = self.bn3(x, training=training)

        if training:
            x = self.dropout(x)

        if self.use_se:
            x = self.se(x)

        output = getattr(tf.nn, self.activation)(tf.keras.layers.add([residual, x]))
        return output

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "filter_num": self.filter_num,
                "dropout_rate": self.dropout_rate,
                "stride": self.stride,
                "activation": self.activation,
                "bn": self.bn,
                "groups": self.groups,
                "kernel_initializer": self.kernel_initializer,
                "kernel_regularizer": self.kernel_regularizer,
                "divide_factor": self.divide_factor,
                "downsampling": self._downsampling_type,
                "use_se": self.use_se,
                "se_ratio": self.se_ratio,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


# Adapted from:
# https://github.com/tensorflow/models/blob/v2.15.0/official/vision/modeling/layers/nn_layers.py
class SqueezeExcitation(tf.keras.layers.Layer):
    """Squeeze-and-excitation block. Used internally by BottleNeck when use_se=True."""

    def __init__(
        self,
        in_filters,
        out_filters,
        se_ratio=1,
        divisible_by=8,
        use_3d_input=True,
        kernel_initializer="VarianceScaling",
        kernel_regularizer=None,
        bias_regularizer=None,
        activation="relu",
        gating_activation="sigmoid",
        round_down_protect=True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._in_filters = in_filters
        self._out_filters = out_filters
        self._se_ratio = se_ratio
        self._divisible_by = divisible_by
        self._round_down_protect = round_down_protect
        self._use_3d_input = use_3d_input
        self._activation = activation
        self._gating_activation = gating_activation
        self._kernel_initializer = kernel_initializer
        self._kernel_regularizer = kernel_regularizer
        self._bias_regularizer = bias_regularizer

        if tf.keras.backend.image_data_format() == "channels_last":
            self._spatial_axis = [1, 2, 3] if use_3d_input else [1, 2]
        else:
            self._spatial_axis = [2, 3, 4] if use_3d_input else [2, 3]

        self._activation_fn = getattr(tf.nn, activation)
        self._gating_activation_fn = getattr(tf.nn, gating_activation)

    @staticmethod
    def make_divisible(
        value: float,
        divisor: int,
        min_value=None,
        round_down_protect: bool = True,
    ) -> int:
        """Correction from the original implementations"""
        if min_value is None:
            min_value = divisor
        new_value = max(min_value, int(value + divisor / 2) // divisor * divisor)
        if round_down_protect and new_value < 0.9 * value:
            new_value += divisor
        return int(new_value)

    def build(self, input_shape):
        """Build layer layers"""
        num_reduced_filters = SqueezeExcitation.make_divisible(
            max(1, int(self._in_filters * self._se_ratio)),
            divisor=self._divisible_by,
            round_down_protect=self._round_down_protect,
        )
        self._se_reduce = tf.keras.layers.Conv2D(
            filters=num_reduced_filters,
            kernel_size=1,
            strides=1,
            padding="same",
            use_bias=True,
            kernel_initializer=self._kernel_initializer,
            kernel_regularizer=self._kernel_regularizer,
            bias_regularizer=self._bias_regularizer,
        )
        self._se_expand = tf.keras.layers.Conv2D(
            filters=self._out_filters,
            kernel_size=1,
            strides=1,
            padding="same",
            use_bias=True,
            kernel_initializer=self._kernel_initializer,
            kernel_regularizer=self._kernel_regularizer,
            bias_regularizer=self._bias_regularizer,
        )
        super().build(input_shape)

    def call(self, inputs):
        """Run forward pass."""

        x = tf.reduce_mean(inputs, self._spatial_axis, keepdims=True)
        x = self._activation_fn(self._se_reduce(x))
        x = self._gating_activation_fn(self._se_expand(x))
        return x * inputs

    def get_config(self):
        """Get default configuration"""
        config = super().get_config()
        config.update(
            {
                "in_filters": self._in_filters,
                "out_filters": self._out_filters,
                "se_ratio": self._se_ratio,
                "divisible_by": self._divisible_by,
                "use_3d_input": self._use_3d_input,
                "kernel_initializer": self._kernel_initializer,
                "kernel_regularizer": self._kernel_regularizer,
                "bias_regularizer": self._bias_regularizer,
                "activation": self._activation,
                "gating_activation": self._gating_activation,
                "round_down_protect": self._round_down_protect,
            }
        )
        return config


# ---------------------------------------------------------------------------
# Layer factory
# ---------------------------------------------------------------------------

_CONV_BLOCKS = {
    "Plain": Plain,
    "BottleNeck": BottleNeck, # The one we used for NeuroFM
    "Residual": Residual,
}


def add_conv_layer(layer_name: str, x, **kwargs):
    """
    Instantiate and apply a conv block by name.

    Parameters
    ----------
    layer_name : str
        One of 'Plain', 'BottleNeck', 'Residual'.
    x : tf.Tensor
        Input tensor.
    **kwargs
        Passed directly to the layer constructor.

    Returns
    -------
    tf.Tensor
    """
    if layer_name not in _CONV_BLOCKS:
        raise ValueError(
            f"Unknown conv block '{layer_name}'. "
            f"Valid options: {list(_CONV_BLOCKS.keys())}"
        )
    return _CONV_BLOCKS[layer_name](**kwargs)(x)
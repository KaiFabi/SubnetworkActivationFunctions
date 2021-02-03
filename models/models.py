"""Module to build MLP and CNN models in Tensorflow.

This module implements basic versions of MLPs and CNNs in Tensorflow. Both classes expect a configuration dictionary.
See also config.yml.

    Typical usage example:

    network = MLP(conf)

    OR

    network = CNN(conf)
"""
import tensorflow as tf
from models.subnet import SubNetwork


class MLP(object):
    """Class to create multilayer perceptron models.
    """

    def __init__(self, conf):
        self.conf = conf

        image_width = conf["dataset"]["image_width"]
        image_height = conf["dataset"]["image_height"]
        n_channels = conf["dataset"]["n_channels"]

        self.input_shape = (image_height * image_width * n_channels)

        self.n_classes = conf["dataset"]["n_classes"]
        self.layers_dense = conf["network"]["units_dense"]
        self.layers_subnet = conf["subnetwork"]["units"]
        self.use_subnet = conf["subnetwork"]["use_subnet"]

        activation_function = conf["network"]["activation_function"]
        if activation_function == "relu":
            self.activation_function = tf.nn.relu
            scale = 2.0
        elif activation_function == "leaky_relu":
            self.activation_function = tf.nn.leaky_relu
            scale = 2.0 / (1.0 + 0.3**2)
        else:
            raise Exception("Error: activation function not implemented.")

        self.initializer = tf.keras.initializers.VarianceScaling(scale=scale,
                                                                 mode='fan_in',
                                                                 distribution='truncated_normal')

        self.dense_config = dict(kernel_initializer=self.initializer)
        self.subnet_activation_function = conf["subnetwork"]["activation_function"]
        self.dropout_rate = conf["training"]["dropout_rate"]

    def build(self):
        inputs = tf.keras.layers.Input(shape=self.input_shape)
        x = tf.keras.layers.Dropout(rate=0.5)(inputs)

        if self.use_subnet:
            z = tf.keras.layers.Dense(units=self.layers_dense[0], **self.dense_config)(x)
            x = SubNetwork(self.conf)(z)
        else:
            x = tf.keras.layers.Dense(units=self.layers_dense[0], activation=self.activation_function,
                                      **self.dense_config)(x)

        if self.dropout_rate > 0.0:
            x = tf.keras.layers.Dropout(rate=self.dropout_rate)(x)

        for units in self.layers_dense[1:]:
            if self.use_subnet:
                z = tf.keras.layers.Dense(units=units, **self.dense_config)(x)
                x = SubNetwork(self.conf)(z)
            else:
                x = tf.keras.layers.Dense(units=units, activation=self.activation_function, **self.dense_config)(x)

            if self.dropout_rate > 0.0:
                x = tf.keras.layers.Dropout(rate=self.dropout_rate)(x)

        outputs = tf.keras.layers.Dense(self.n_classes)(x)
        model = tf.keras.Model(inputs=inputs, outputs=outputs, name="mlp")
        return model


class CNN(object):
    """Class to create VGG-like convolutional neural networks.
    """

    def __init__(self, conf):
        self.conf = conf

        image_width = conf["dataset"]["image_width"]
        image_height = conf["dataset"]["image_height"]
        n_channels = conf["dataset"]["n_channels"]
        self.input_shape = (image_height, image_width, n_channels)

        self.n_classes = conf["dataset"]["n_classes"]
        self.units_conv = conf["network"]["units_conv"]
        self.units_dense = conf["network"]["units_dense"]

        activation_function = conf["network"]["activation_function"]
        if activation_function == "relu":
            self.activation_function = tf.nn.relu
            scale = 2.0
        elif activation_function == "leaky_relu":
            self.activation_function = tf.nn.leaky_relu
            scale = 2.0 / (1.0 + 0.3**2)
        else:
            raise Exception("Error: activation function not implemented.")

        self.use_subnet = conf["subnetwork"]["use_subnet"]
        self.units_subnet = conf["subnetwork"]["units"]
        self.subnet_activation_function = conf["subnetwork"]["activation_function"]

        self.initializer = tf.keras.initializers.VarianceScaling(scale=scale,
                                                                 mode='fan_in',
                                                                 distribution='truncated_normal')

        self.conf_conv = dict(kernel_size=(3, 3),
                              strides=(1, 1),
                              activation=tf.identity,
                              padding='same',
                              kernel_initializer=self.initializer)

        self.conf_pool = dict(pool_size=(2, 2), padding='same')

        self.conf_dense = dict(kernel_initializer=self.initializer)
        self.dropout_rate = conf["training"]["dropout_rate"]

    def build(self):
        inputs = tf.keras.layers.Input(shape=self.input_shape)
        x = inputs

        # Convolutional part
        for filters in self.units_conv:
            x = self.conv_block(x, filters)
        x = tf.keras.layers.Flatten()(x)

        # Dense part
        for units in self.units_dense:
            x = tf.keras.layers.Dense(units=units, **self.conf_dense)(x)
            if self.use_subnet:
                x = SubNetwork(self.conf)(x)
            else:
                x = self.activation_function(x)

            if self.dropout_rate > 0.0:
                x = tf.keras.layers.Dropout(rate=self.dropout_rate)(x)

        x = tf.keras.layers.Dense(units=self.n_classes, **self.conf_dense)(x)

        model = tf.keras.models.Model(inputs=inputs, outputs=x)
        return model

    def conv_block(self, inputs, filters):
        x = tf.keras.layers.Conv2D(filters=filters, **self.conf_conv)(inputs)
        if self.use_subnet:
            x = SubNetwork(self.conf)(x)
        else:
            x = self.activation_function(x)

        x = tf.keras.layers.Conv2D(filters=filters, **self.conf_conv)(x)
        if self.use_subnet:
            x = SubNetwork(self.conf)(x)
        else:
            x = self.activation_function(x)

        x = tf.keras.layers.MaxPool2D(**self.conf_pool)(x)

        return x

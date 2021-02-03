"""This module implements subnetwork activation functions using the Tensorflow layer class.

The class in this file uses the Tensorflow Keras layer class to implement a basic version
of a subnetwork which can act as a trainable activation functions replacing standard
activation functions by fully connected neural networks.

    Typical usage example:

        subnet = SubNetwork(self.conf)
        x = subnet(z)

        OR

        x = SubNetwork(self.conf)(z)
"""
import tensorflow as tf


class SubNetwork(tf.keras.layers.Layer):
    """Creates trainable activation functions using a fully connected neural network.

        This class replaces standard activation functions with a multilayer perceptron
        and can be used as any other layer.

    Attributes:
        units: A list describing the subnetwork's layout.
        activation_function: The nonlinearity used in the subnetwork.
        scale: Scaling factor for weight initializer.
        kernel: Placeholder for subnetwork's kernels.
        bias: Placeholder for subnetwork's biases.
    """
    def __init__(self, conf):
        super(SubNetwork, self).__init__()

        self.units = conf["subnetwork"]["units"]
        self.scale = 1.0

        activation_function = conf["subnetwork"]["activation_function"]
        if activation_function == "leaky_relu":
            self.activation_function = tf.nn.leaky_relu
            self.scale = 2.0 / (1.0 + 0.3**2)
        elif activation_function == "relu":
            self.activation_function = tf.nn.relu
            self.scale = 2.0
        elif activation_function == "sigmoid":
            self.activation_function = tf.nn.sigmoid
        elif activation_function == "tanh":
            self.activation_function = tf.math.tanh
        elif activation_function == "sin":
            self.activation_function = tf.math.sin
        else:
            raise Exception("Error: activation function not implemented.")

        self.kernel = None
        self.bias = None

    def build(self, _):
        """Initializes the subnetwork's trainable parameters.
        """
        bias_initializer = tf.keras.initializers.Zeros()
        kernel_initializer = tf.keras.initializers.VarianceScaling(scale=self.scale, mode='fan_in',
                                                                   distribution='truncated_normal')

        self.bias = [self.add_weight(shape=(i,), initializer=bias_initializer, trainable=True, name="bias")
                     for i in self.units[1:]]

        self.kernel = [self.add_weight(shape=(i, j), initializer=kernel_initializer, trainable=True, name="kernel")
                       for i, j in zip(self.units[:-1], self.units[1:])]

    def call(self, input_tensor):
        """Builds subnetwork's graph.

        Args:
            input_tensor: Tensor consisting of the network's pre-activations

        Returns:
            x: Tensor with same dimensions as input_tensor

        """
        # Reshape input into one-dimensional tensor
        x = tf.reshape(tensor=input_tensor, shape=(-1, 1))

        # Feedforward pre-activation through network
        for kernel, bias in zip(self.kernel[:-1], self.bias[:-1]):
            x = tf.matmul(x, kernel) + bias
            x = self.activation_function(x)
        x = tf.matmul(x, self.kernel[-1]) + self.bias[-1]

        # Reshape back to input shape and return
        x = tf.reshape(tensor=x, shape=tf.shape(input_tensor))
        return x

    def get_config(self):
        config = super(SubNetwork, self).get_config()
        config.update({"units": self.units})
        return config

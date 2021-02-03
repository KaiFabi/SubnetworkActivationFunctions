"""Helper functions for training and evaluation of subnetworks.
"""

import tensorflow.keras.backend as K
import tensorflow as tf
import numpy as np


def cyclic_learning_rate_scheduler(epoch, lr):
    """Implements a cyclic learning rate scheduler.

    Args:
        epoch: Current epoch.
        lr: Current learning rate. Not used here.

    Returns:
        lr: New learning rate.

    """
    return lr
    #lr_min = 1e-8
    #lr_max = 1e-3
    #cycles_per_epoch = 0.1
    #return (lr_max - lr_min) * tf.math.abs(tf.math.sin(np.pi * cycles_per_epoch * epoch)) + lr_min


def get_preactivation_statistics(model, x_sample):
    """Computes statistics of pre-activations that go into the subnetwork.

    Args:
        model: Tensorflow Keras model.
        x_sample: Samples from test dataset.

    Returns:
        stats: Dictionary with pre-activation statistics.

    """
    # Input placeholder
    inputs = model.input

    # Store all layer outputs
    outputs = [layer.output for layer in model.layers if "conv2d" in layer.name or "dense" in layer.name]

    # Skip last layer output since there is no subnetwork involved
    outputs = outputs[:-1]

    # Build graph
    f = K.function(inputs=inputs, outputs=outputs)

    # Get pre-activations
    pre_activations = f([x_sample])
    stats = {output.name: {"avg": z.mean(), "std": z.std(), "min": z.min(), "max": z.max()}
             for output, z in zip(outputs, pre_activations)}

    return stats


def get_subnet_graphs(model, stats, sigma=3.0, n_points=1000):
    """Uses subnetworks of model to create corresponding graph.

    Args:
        model: Tensorflow Keras model.
        stats: Dictionary that contains avg, std, min, and max of subnetworks' pre-activations.
        sigma: Scalar defining domain of input values x.
        n_points: Number of points x to compute f(x).

    Returns:
        f: Dictionary containing graphs of subnetworks.
    """
    # Get model's subnetworks
    subnetworks = get_subnets(model)

    # Compute subnetwork's graph
    f = {layer_name: {"x": None, "y": None} for layer_name in stats.keys()}
    for subnetwork, (layer_name, value) in zip(subnetworks, stats.items()):
        x_avg = value["avg"]
        x_std = value["std"]
        x_min = x_avg - sigma * x_std
        x_max = x_avg + sigma * x_std
        x = np.linspace(x_min, x_max, num=n_points)
        y = subnetwork.predict(x=x)[:, 0]
        f[layer_name]["x"] = x
        f[layer_name]["y"] = y

    return f


def get_subnets(model):
    """Function to extract subnetwork from Tensorflow Keras model.

    Args:
        model: Tensorflow Keras model.

    Returns:
        subnetworks: List of Tensorflow Keras models.
    """
    n_subnetworks = comp_num_subnets(model)
    subnetworks = [tf.keras.models.Sequential() for _ in range(n_subnetworks)]
    counter = 0
    for i, layer in enumerate(model.layers):
        if 'sub_network' in layer.name:
            subnetworks[counter].add(model.get_layer(index=i))
            counter += 1
    return subnetworks


def comp_num_subnets(model):
    """Counts number of subnetworks in network.

    Args:
        model: Tensorflow Keras model.

    Returns:
        n_subnetworks: Number of subnetworks inside the network.

    Raises:
        Error if model does not contain any subnetworks.
    """
    n_subnetworks = 0
    for layer in model.layers:
        if 'sub_network' in layer.name:
            n_subnetworks += 1

    if n_subnetworks == 0:
        raise Exception("Error: No subnetworks found.")

    return n_subnetworks

import tensorflow as tf
import numpy as np


def normalize(x):
    x_min = np.min(x, keepdims=True)
    x_max = np.max(x, keepdims=True)
    return (x - x_min) / (x_max - x_min)


def rescale(x, a=0.0, b=1.0):
    return a + (b - a) * normalize(x)


def standardize(x, x_avg, x_std):
    return (x - x_avg) / x_std


def data(conf):

    network_type = conf["network"]["type"]
    dataset_name = conf["dataset"]["name"]

    if network_type == "mlp":

        if dataset_name == 'cifar10':
            (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
            x_train, x_test = rescale(x_train), rescale(x_test)
            x_avg = np.mean(x_train, axis=(0, 1, 2))
            x_std = np.std(x_train, axis=(0, 1, 2))
            x_train = standardize(x_train, x_avg, x_std)
            x_test = standardize(x_test, x_avg, x_std)

        elif dataset_name == 'fashion_mnist':
            (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
            x_train, y_train = rescale(x_train).reshape(-1, 28**2), y_train.astype(np.float)
            x_test, y_test = rescale(x_test).reshape(-1, 28**2), y_test.astype(np.float)

        elif dataset_name == 'mnist':
            (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
            x_train, y_train = rescale(x_train).reshape(-1, 28**2), y_train.astype(np.float)
            x_test, y_test = rescale(x_test).reshape(-1, 28**2), y_test.astype(np.float)

        else:
            raise Exception('Error: No such dataset defined.')

    elif network_type == 'cnn':

        if dataset_name == "cifar10":
            (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
            x_train, x_test = rescale(x_train), rescale(x_test)
            x_avg = np.mean(x_train, axis=(0, 1, 2))
            x_std = np.std(x_train, axis=(0, 1, 2))
            x_train = standardize(x_train, x_avg, x_std)
            x_test = standardize(x_test, x_avg, x_std)

        elif dataset_name == "fashion_mnist":
            (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
            x_train, y_train = np.expand_dims(rescale(x_train), axis=-1), y_train.astype(np.float)
            x_test, y_test = np.expand_dims(rescale(x_test), axis=-1), y_test.astype(np.float)

        elif dataset_name == "mnist":
            (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
            x_train, y_train = np.expand_dims(rescale(x_train), axis=-1), y_train.astype(np.float)
            x_test, y_test = np.expand_dims(rescale(x_test), axis=-1), y_test.astype(np.float)

        else:
            raise Exception('No such dataset defined.')

    else:
        raise Exception('No such network defined.')

    return x_train, y_train, x_test, y_test

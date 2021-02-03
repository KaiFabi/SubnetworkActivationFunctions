"""This module loads pre-trained Tensorflow models and visualizes their performance and
   their subnetwork's graph if available. It ain't pretty but it works.

This file consists of two main methods. plot_subnets_stats() visualizes a network's subnetworks' graphs.
plot_stats() compares the performance of network with and without subnetworks as activation functions by
visualizing loss and accuracy.
"""

import os
import yaml
import numpy as np
import matplotlib.pyplot as plt

from data.data import data
from models.models import CNN, MLP
from utils.utils import get_subnet_graphs, get_preactivation_statistics

from tensorflow.python.summary.summary_iterator import summary_iterator

# Make GPU invisible to Tensorflow.
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


def plot_subnets_stats(conf):
    """Visualizes the graphs of subnetworks.

    Args:
        conf: Dictionary consisting of configuration parameters.
    """

    network_type = conf["network"]["type"]
    if network_type == "cnn":
        network = CNN(conf)
    elif network_type == "mlp":
        network = MLP(conf)

    model = network.build()

    _, _, x_test, _ = data(conf)
    x_test = x_test[:1000]

    # Get directories of all models
    dir_names = next(os.walk(conf["paths"]["model"]))[1]
    model_dirs = ["{}{}{}".format(conf["paths"]["model"], dir_name, "/model.ckpt")
                  for dir_name in dir_names if dir_name.split("/")[-1].endswith("1")]
    print("{} models found.".format(len(model_dirs)))

    # Determine avg and std of subnet activations
    subnets_statistics = list()
    for model_dir in model_dirs:
        model.load_weights(model_dir).expect_partial()
        stats = get_preactivation_statistics(model, x_test)
        subnets_statistics.append(stats)

    # Extract layer names
    layer_names = [layer_name for layer_name in subnets_statistics[0].keys()]
    print("{} subnets found.".format(len(layer_names)))

    stats = {layer_name: {"avg": None, "std": None} for layer_name in layer_names}

    # Compute average stats of all subnets
    for layer_name in layer_names:
        avg = 0.0
        std = 0.0
        for subnet in subnets_statistics:
            avg += subnet[layer_name]["avg"]
            std += subnet[layer_name]["std"]
        stats[layer_name]["avg"] = avg / len(subnets_statistics)
        stats[layer_name]["std"] = std / len(subnets_statistics)

    # Extract graphs from subnets from all models
    graphs_dict = {layer_name: {"x": list(), "y": list()} for layer_name in layer_names}
    for model_dir in model_dirs:
        model.load_weights(model_dir).expect_partial()
        graphs = get_subnet_graphs(model, stats)
        for layer_name in layer_names:
            graphs_dict[layer_name]["x"].append(graphs[layer_name]["x"])
            graphs_dict[layer_name]["y"].append(graphs[layer_name]["y"])

    stats = {layer_name: {"x": None, "y_avg": None, "y_std": None} for layer_name in layer_names}

    for layer_name in layer_names:
        stats[layer_name]["x"] = np.mean(graphs_dict[layer_name]["x"], axis=0)  # not necessary
        stats[layer_name]["y_avg"] = np.mean(graphs_dict[layer_name]["y"], axis=0)
        stats[layer_name]["y_std"] = np.std(graphs_dict[layer_name]["y"], axis=0)

    # Plot results
    nrows = 2
    ncols = 4
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(2*ncols, 2*nrows))
    for i, (layer_name, ax) in enumerate(zip(layer_names, axes.flatten())):
        x = stats[layer_name]["x"]
        y_avg = stats[layer_name]["y_avg"]
        y_std = stats[layer_name]["y_std"]
        x_all = np.squeeze(graphs_dict[layer_name]["x"]).T
        y_all = np.squeeze(graphs_dict[layer_name]["y"]).T
        ax.plot(x_all, y_all, linewidth=0.3, alpha=0.4, color="green")
        ax.plot(x, y_avg, linewidth=1.0, color="green")
        ax.fill_between(x, y_avg - y_std, y_avg + y_std, alpha=0.2, color="green")
        ax.grid(True, alpha=0.5, linestyle='dotted')
        ax.set_title("$s^{{({layer})}}$".format(layer=str(i+1)))
    plt.tight_layout()
    plt.savefig('{}subnetworks_{}.png'.format(conf["paths"]["results"], "test"), dpi=100)
    plt.close(fig)


def plot_stats(conf):
    """This verbose method visualizes loss and accuracy of networks equipped with and without subnetworks.

    Args:
        conf: Dictionary consisting of configuration parameters.
    """

    # Get paths of all event files
    stats_dir = conf["paths"]["stats"]
    event_file_paths = list()
    for root, dirs, files in os.walk(stats_dir):
        if "train" in root or "valid" in root:
            for file in files:
                if file.endswith(".v2"):
                    event_file_paths.append(os.path.join(root, files[0]))

    # Create dictionary with statistics for every model
    model_names = next(os.walk(conf["paths"]["model"]))[1]
    stats = dict()
    for model_name in model_names:
        stats[model_name] = dict(train=dict(epoch_loss=[], epoch_accuracy=[]),
                                 validation=dict(epoch_loss=[], epoch_accuracy=[]))

    # Fill dictionary
    for event_file_path in event_file_paths:
        event_info = event_file_path.split("/")
        event_tag = event_info[-2]
        model_name = event_info[-3]
        for event in summary_iterator(event_file_path):
            for value in event.summary.value:
                if value.tag == "epoch_loss" or value.tag == "epoch_accuracy":
                    stats[model_name][event_tag][value.tag].append(value.simple_value)

    # Create summary of statistics
    summary = dict()
    network_types = ["0", "1"]
    for network_type in network_types:
        summary[network_type] = dict(train=dict(epoch_loss=dict(raw=[], avg=[], std=[]),
                                                epoch_accuracy=dict(raw=[], avg=[], std=[])),
                                     validation=dict(epoch_loss=dict(raw=[], avg=[], std=[]),
                                                     epoch_accuracy=dict(raw=[], avg=[], std=[])))

    event_types = ["train", "validation"]
    evaluation_types = ["epoch_loss", "epoch_accuracy"]

    for model_name in model_names:
        for event_type in event_types:
            for evaluation_type in evaluation_types:
                x = stats[model_name][event_type][evaluation_type]
                if len(x) == conf["training"]["epochs"]:
                    if model_name.endswith("0"):
                        summary["0"][event_type][evaluation_type]["raw"].append(x)
                    elif model_name.endswith("1"):
                        summary["1"][event_type][evaluation_type]["raw"].append(x)

    for network_type in network_types:
        for event_type in event_types:
            for evaluation_type in evaluation_types:
                x = summary[network_type][event_type][evaluation_type]["raw"]
                summary[network_type][event_type][evaluation_type]["avg"] = np.mean(x, axis=0)
                summary[network_type][event_type][evaluation_type]["std"] = np.std(x, axis=0)
                del summary[network_type][event_type][evaluation_type]["raw"]

    args = dict(linewidth=0.3, alpha=0.4, color=None)
    for evaluation_type in evaluation_types:
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(8, 4))
        for ax, event_type in zip(axes, event_types):
            for model, statistics in stats.items():
                if model.endswith("0"):
                    args["color"] = "red"
                elif model.endswith("1"):
                    args["color"] = "green"
                ax.plot(statistics[event_type][evaluation_type], **args)

            for network_type in network_types:
                if network_type == "0":
                    args["color"] = "red"
                    label = "Leaky ReLU"
                elif network_type == "1":
                    args["color"] = "green"
                    label = "Subnetwork"
                y_avg = summary[network_type][event_type][evaluation_type]["avg"]
                y_std = summary[network_type][event_type][evaluation_type]["std"]
                x = list(range(len(y_avg)))
                ax.plot(y_avg, linewidth=1.0, color=args["color"], label=label)
                ax.legend()
                ax.fill_between(x, y_avg-y_std, y_avg+y_std, alpha=0.2, color=args["color"])

                prototype = "{} {} {} {:.4f} +- {:.4f}".format(network_type,
                                                             ("train" if event_type == "train" else "valid"),
                                                             ("loss" if evaluation_type == "epoch_loss" else "accu"),
                                                             y_avg[-1], y_std[-1])
                print(prototype)

            ax.grid(True, alpha=0.5, linestyle='dotted')
            if event_type == "train":
                event_name = "Training"
            elif event_type == "validation":
                event_name = "Validation"
            if evaluation_type == "epoch_accuracy":
                evaluation_name = "accuracy"
            elif evaluation_type == "epoch_loss":
                evaluation_name = "loss"
            ax.set_title(event_name + " " + evaluation_name)
            ax.set_xlabel("Epochs")
        plt.tight_layout()
        plt.savefig(conf["paths"]["results"] + evaluation_type + ".png", dpi=100)
        plt.close(fig)


def main():
    conf = yaml.safe_load(open("config.yml"))
    plot_subnets_stats(conf)
    plot_stats(conf)


if __name__ == '__main__':
    main()

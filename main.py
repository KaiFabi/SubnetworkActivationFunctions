"""This module tests MLPs and CNNs with subnetworks that replace activation functions.
"""
import yaml
import datetime
import tensorflow as tf

from pathlib import Path
from data.data import data
from models.models import CNN, MLP
from utils.utils import comp_num_subnets, cyclic_learning_rate_scheduler

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)


def run_mlp(conf):

    x_train, y_train, x_test, y_test = data(conf)
    network = MLP(conf)
    model = network.build()
    model.summary()

    if conf["subnetwork"]["use_subnet"]:
        print("Number of subnetworks: {}".format(comp_num_subnets(model)))

    tf.keras.utils.plot_model(model,
                              conf["paths"]["results"] + "model.png",
                              show_shapes=True,
                              show_layer_names=True,
                              expand_nested=True)

    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  optimizer=tf.keras.optimizers.Adam(learning_rate=conf["training"]["learning_rate"]),
                  metrics=['accuracy'])

    date_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    appendix = "1" if conf["subnetwork"]["use_subnet"] else "0"
    stats_dir = conf["paths"]["stats"] + date_time + "_" + appendix
    Path(stats_dir).mkdir(parents=True, exist_ok=True)
    model_dir = conf["paths"]["model"] + date_time + "_" + appendix
    Path(model_dir).mkdir(parents=True, exist_ok=True)

    callback_learning_rate = tf.keras.callbacks.LearningRateScheduler(cyclic_learning_rate_scheduler)
    callback_tensorboard = tf.keras.callbacks.TensorBoard(log_dir=stats_dir, histogram_freq=1)
    callback_checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=model_dir + "/model.ckpt",
                                                             save_weights_only=True,
                                                             monitor='val_accuracy',
                                                             mode='max',
                                                             save_best_only=True,
                                                             save_freq='epoch',
                                                             verbose=1)

    epochs = conf["training"]["epochs"]
    batch_size = conf["training"]["batch_size"]
    model.fit(x=x_train,
              y=y_train,
              epochs=epochs,
              batch_size=batch_size,
              validation_data=(x_test, y_test),
              callbacks=[callback_learning_rate, callback_tensorboard, callback_checkpoint],
              verbose=2)

    tf.keras.backend.clear_session()


def run_cnn(conf):

    x_train, y_train, x_test, y_test = data(conf)

    network = CNN(conf)
    model = network.build()
    model.summary()

    if conf["subnetwork"]["use_subnet"]:
        print("Number of subnetworks: {}".format(comp_num_subnets(model)))

    tf.keras.utils.plot_model(model,
                              conf["paths"]["results"] + "model.png",
                              show_shapes=True,
                              show_layer_names=True,
                              expand_nested=True)

    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  optimizer=tf.keras.optimizers.Adam(learning_rate=conf["training"]["learning_rate"]),
                  metrics=['accuracy'])

    datagen = tf.keras.preprocessing.image.ImageDataGenerator(horizontal_flip=True,
                                                              rotation_range=10.0,
                                                              width_shift_range=0.2,
                                                              height_shift_range=0.2,
                                                              shear_range=10.0,
                                                              zoom_range=0.1)

    date_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    appendix = "1" if conf["subnetwork"]["use_subnet"] else "0"
    stats_dir = conf["paths"]["stats"] + date_time + "_" + appendix
    Path(stats_dir).mkdir(parents=True, exist_ok=True)
    model_dir = conf["paths"]["model"] + date_time + "_" + appendix
    Path(model_dir).mkdir(parents=True, exist_ok=True)

    callback_tensorboard = tf.keras.callbacks.TensorBoard(log_dir=stats_dir, histogram_freq=1)
    callback_checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=model_dir + "/model.ckpt",
                                                             save_weights_only=True,
                                                             monitor='val_accuracy',
                                                             mode='max',
                                                             save_best_only=True,
                                                             save_freq='epoch',
                                                             verbose=1)

    batch_size = conf["training"]["batch_size"]
    epochs = conf["training"]["epochs"]
    model.fit(datagen.flow(x_train, y_train, batch_size=batch_size),
              steps_per_epoch=len(x_train) // batch_size,
              epochs=epochs,
              validation_data=(x_test, y_test),
              validation_steps=len(x_test) // batch_size,
              callbacks=[callback_tensorboard, callback_checkpoint],
              verbose=2)

    tf.keras.backend.clear_session()


def test_mlp_network(conf):

    for _ in range(10):
        conf["subnetwork"]["use_subnet"] = True
        run_mlp(conf)
        conf["subnetwork"]["use_subnet"] = False
        run_mlp(conf)


def test_cnn_network(conf):

    for _ in range(10):
        conf["subnetwork"]["use_subnet"] = True
        run_cnn(conf)
        conf["subnetwork"]["use_subnet"] = False
        run_cnn(conf)


def main():
    conf = yaml.safe_load(open("config.yml"))

    if conf["network"]["type"] == "mlp":
        test_mlp_network(conf)
    elif conf["network"]["type"] == "cnn":
        test_cnn_network(conf)


if __name__ == '__main__':
    main()

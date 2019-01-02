import os
import logging

import pandas as pd
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from scipy import constants as const
from generic import prepare
from black_body import conf

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s : [%(levelname)s] : %(name)s : %(message)s')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def planck(wav, T):
    a = 2.0 * const.h * const.c ** 2
    b = const.h * const.c / (wav * const.k * T)
    intensity = a / ((wav ** 5) * (np.exp(b) - 1.0))
    return intensity


def get_data():
    wavelengths = np.arange(1e-9, 3e-6, 1e-9)
    intensity4000 = planck(wavelengths, 4000.)
    intensity4000 = intensity4000/max(intensity4000)
    wavelengths = wavelengths / max(wavelengths)

    return pd.DataFrame({
        "wave": wavelengths,
        "flux": intensity4000
    })


def get_sample(test_size):
    data = get_data()
    data["flux"] = data["flux"] / max(data["flux"])
    data["wave"] = data["wave"] / max(data["wave"])
    return prepare.split_training_set(
        x=np.array(data["wave"]).reshape(len(data), 1),
        y=np.array(data["flux"]).reshape(len(data), 1), test_size=test_size)


class MlpNet(object):
    def __init__(self, train_samples, test_samples):
        self._logger = logging.getLogger(MlpNet.__name__)
        self._logger.info("craeting neural net model with following paramters: {}".format(conf.serialize_nn_params()))

        self.tf_x = tf.placeholder(tf.float32, [None, conf.input_size])
        self.tf_y = tf.placeholder(tf.float32)

        self.hidden_1_layer = {
            "weights": tf.Variable(tf.random_normal([conf.input_size, conf.n_nodes_hl1])),
            "biases": tf.Variable(tf.random_normal([conf.n_nodes_hl1]))
        }

        self.hidden_2_layer = {
            "weights": tf.Variable(tf.random_normal([conf.n_nodes_hl1, conf.n_nodes_hl2])),
            "biases": tf.Variable(tf.random_normal([conf.n_nodes_hl2]))
        }

        self.output_layer = {
            "weights": tf.Variable(tf.random_normal([conf.n_nodes_hl2, 1])),
            "biases": tf.Variable(tf.random_normal([1]))}

        self.nn_model_output = None

        self.train_samples = train_samples
        self.test_samples = test_samples

    def evaluate_nn_model(self):
        self._logger.info("evaluating mlp neural model")

        layer_1 = tf.add(
            tf.matmul(self.tf_x, self.hidden_1_layer["weights"]),
            self.hidden_1_layer["biases"]
        )
        # treshold function
        layer_1 = tf.nn.sigmoid(layer_1)

        layer_2 = tf.add(
            tf.matmul(layer_1, self.hidden_2_layer["weights"]),
            self.hidden_2_layer["biases"]
        )
        # treshold function
        layer_2 = tf.nn.sigmoid(layer_2)

        self.nn_model_output = tf.add(
            tf.matmul(layer_2, self.output_layer["weights"]),
            self.output_layer["biases"]
        )
        self._logger.info("evaluation of mlp neural model done, saved onto property `nn_model_output`")

    def train(self):
        self._logger.info("starting training of mlp neural model")
        if self.nn_model_output is None:
            raise ValueError("evaluate mlp first. run self.evaluate_nn_model")

        prediction = self.nn_model_output
        loss_fn = tf.losses.mean_squared_error(predictions=prediction, labels=self.tf_y)
        optimizer = tf.train.AdamOptimizer(conf.learning_rate).minimize(loss_fn)

        self._logger.info("eval tensorflow session contextual manager")
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            for epoch in range(conf.train_epochs):
                epoch_loss, i = 0, 0
                np.random.shuffle(self.train_samples)

                while i < len(self.train_samples):
                    slice_start = i
                    slice_stop = i + conf.batch_size

                    next_batch = self.train_samples[slice_start: slice_stop]
                    zipped = list(zip(*next_batch))
                    x_feed, y_feed = zipped[0], zipped[1]

                    _, c = sess.run([optimizer, loss_fn], feed_dict={self.tf_x: x_feed, self.tf_y: y_feed})
                    epoch_loss += c
                    i += conf.batch_size
                if epoch % 100 == 0:
                    self._logger.info('epoch {} completed out of {}, loss: {}'
                                      ''.format(epoch, conf.train_epochs, epoch_loss))

            def evaluate_pecission():
                train = list(zip(*self.train_samples))
                x = train[0]
                y = train[1]

                x_tst = np.arange(0, 1, 0.01)
                x_tst = x_tst.reshape(len(x_tst), 1)

                output = sess.run(self.nn_model_output, feed_dict={self.tf_x: x_tst})

                plt.scatter(x, y, s=0.1, marker='x', c='r')
                plt.scatter(x_tst, output, s=0.1, marker='o', c='b')
                plt.show()

            evaluate_pecission()


if __name__ == "__main__":
    # plot_data()
    t_data = get_sample(test_size=0.1)
    mlp = MlpNet(*t_data)
    mlp.evaluate_nn_model()
    mlp.train()

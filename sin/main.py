import os
import time
import logging
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from sin import conf


logging.basicConfig(level=logging.DEBUG, format='%(asctime)s : [%(levelname)s] : %(name)s : %(message)s')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

logger = logging.getLogger('sin')

def seed(s=None):
    init_value = int(time.time()) if s is None else s
    logging.debug("setting seed with initial value {}".format(init_value))
    np.random.seed(init_value)


def get_samples():
    x = np.arange(conf.sin_start, conf.sin_end, 0.001)
    x = x.reshape(len(x), 1)
    y = np.sin(x)

    train_samples = np.array(list(zip(x, y)))
    np.random.shuffle(train_samples)

    all_entities = len(x)
    test_size = 0.05

    test_samples = train_samples[:int(all_entities * test_size)]
    train_samples = train_samples[int(all_entities * test_size):]

    test_zipped = list(zip(*test_samples))
    test_X, test_y = test_zipped[0], test_zipped[1]
    test_samples = np.array(list(zip(test_X, test_y)))
    return train_samples, test_samples


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
        layer_1 = tf.nn.tanh(layer_1)
        layer_2 = tf.add(
            tf.matmul(layer_1, self.hidden_2_layer["weights"]),
            self.hidden_2_layer["biases"]
        )
        # treshold function
        layer_2 = tf.nn.tanh(layer_2)

        self.nn_model_output = tf.add(
            tf.matmul(layer_2, self.output_layer["weights"]),
            self.output_layer["biases"]
        )

        self._logger.info("evaluation of mlp neural model done, saved onto property `nn_model_output`")

    def train(self):
        self._logger.info("starting training of mlp neural model")

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
                x = np.arange(0, 2 * np.pi, 0.001)
                x = x.reshape(len(x), 1)
                y = np.sin(x)
                output = sess.run(self.nn_model_output, feed_dict={self.tf_x: x})

                plt.scatter(x, y, s=0.1, marker='x', c='r')
                plt.scatter(x, output, s=0.1, marker='o', c='b')
                plt.show()

            evaluate_pecission()


if __name__ == "__main__":
    seed()
    mlp = MlpNet(*get_samples())
    mlp.evaluate_nn_model()
    mlp.train()

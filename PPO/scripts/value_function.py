import tensorflow as tf
import numpy as np
from neural_net import Network

class Value_function:
    def __init__(self, observation_space, learning_rate):
        self.value_pl = tf.placeholder(tf.float32, [None], name='Values')
        self.learning_rate = learning_rate

        self.network = Network()
        self.network.inference_value(observation_space)

        self.loss = self.loss()
        self.optimizer = self.optimizer()

    def loss(self):
        return tf.losses.mean_squared_error(self.network.predict, self.value_pl)

    def optimizer(self):
        opt = tf.train.AdamOptimizer(self.learning_rate)
        training_function = opt.minimize(self.loss, var_list=self.network.variables)
        return training_function

    def fill_feed_dict(self, observations, gradients):
        """
         Fills the feed_dict that will be fed to the placeholders
        Args:
         observations: the current state
        Returns:
         feed_dict: the filled feed dict
        """
        input_feed = observations
        gradients_feed = gradients
        feed_dict = {
            self.input_pl: input_feed,
            self.gradients_pl: gradients_feed
        }
        return feed_dict

    def predict(self, observation, predictor):
        gradients = np.array([np.zeros(self.action_space)])
        feed_dict = self.fill_feed_dict(observation, gradients)
        prediction = self.sess.run(predictor,feed_dict=feed_dict)
        return prediction

    def train(self, observations, gradients):
        feed_dict = self.fill_feed_dict(observations, gradients)
        self.sess.run(self.optimizer, feed_dict=feed_dict)

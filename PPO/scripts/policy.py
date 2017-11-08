import tensorflow as tf
import numpy as np
from neural_net import Network

class Policy:
    def __init__(self, observation_space, action_space, epsilon, learning_rate):
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.adv_pl = tf.placeholder(tf.float32, [None], name='Advantages')
        self.action_pl = tf.placeholder(tf.float32, [None, action_space], name='Actions')

        self.network = Network()
        self.network_old = Network()

        self.network.inference_policy(observation_space, action_space)
        self.network_old.inference_policy(observation_space, action_space)

        self.surrogate = self.surrogate()
        self.optimizer = self.optimizer()

    def surrogate(self):
        r = self.network.mvn.prob(self.action_pl) / self.network_old.mvn.prob(self.action_pl)
        surr1 = r * self.adv_pl
        surr2 = tf.clip_by_value(r, 1.0 - self.epsilon, 1.0 + self.epsilon) * self.adv_pl
        return -tf.reduce_mean(tf.minimum(surr1, surr2))

    def optimizer(self):
        with tf.name_scope('PolicyOptimizer'):
            opt = tf.train.AdamOptimizer(self.learning_rate)
            training_function = opt.minimize(self.surrogate, var_list=self.network.variables)
            return training_function

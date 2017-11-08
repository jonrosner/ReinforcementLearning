import tensorflow as tf
import numpy as np
from tensorflow.contrib.distributions import MultivariateNormalDiag

class Network:
    def inference_value(self, observation_space):
        """
         Creates a neural-network value function approximator
        Args:
         observation_space: observation space of the environment
        Returns:
         Nothing, the network is usable only after calling this method
        """
        self.variables = tf.trainable_variables()
        self.input_pl = tf.placeholder(tf.float32, [None, observation_space], name='Input_PL')
        #2 hidden layer with 100 neurons each
        net = tf.layers.dense(self.input_pl, 100, activation=tf.nn.tanh, kernel_initializer=tf.random_normal_initializer(stddev=.1))
        net = tf.layers.dense(net, 100, activation=tf.nn.tanh, kernel_initializer=tf.random_normal_initializer(stddev=.1))
        net = tf.layers.dense(net, 1, kernel_initializer=tf.random_normal_initializer(stddev=.01))
        self.predict = tf.squeeze(net, axis=[1])
        self.variables = tf.trainable_variables()[len(self.variables):]

    def inference_policy(self, observation_space, action_space):
        """
         Creates a neural-network policy approximator
        Args:
         observation_space: observation space of the environment
         action_space: action space of the environment
        Returns:
         Nothing, the network is usable only after calling this method
        """
        self.variables = tf.trainable_variables()
        self.input_pl = tf.placeholder(tf.float32, [None, observation_space], name='Input_PL')
        #2 hidden layers with 100 neurons each
        net = tf.layers.dense(self.input_pl, 100, activation=tf.nn.tanh, kernel_initializer=tf.random_normal_initializer(stddev=.1))
        net = tf.layers.dense(net, 100, activation=tf.nn.tanh, kernel_initializer=tf.random_normal_initializer(stddev=.1))
        mean = tf.layers.dense(net, action_space, kernel_initializer=tf.random_normal_initializer(stddev=.01))
        self.std = tf.Variable(np.ones(action_space).astype(np.float32))
        self.mvn = MultivariateNormalDiag(mean, self.std)
        self.sample = self.mvn.sample()
        self.variables = tf.trainable_variables()[len(self.variables):]

    def copy_to(self, target_network):
        """
         Operations to copy from self to target
        Args:
         target_network: network to be copied into
        Returns:
         copy_ops: tf-operations (have to be run inside a tf.Session)
        """
        v1 = self.variables
        v2 = target_network.variables
        copy_ops = [v2[i].assign(v1[i]) for i in range(len(v1))]
        return copy_ops

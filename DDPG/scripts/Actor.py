import tensorflow as tf
import numpy as np
from NeuralNet import NeuralNet

class Actor:
    def __init__(self, sess, observation_space, action_space, LEARNING_RATE, NET_SIZE, TAU, action_scale):
        self.sess = sess
        self.observation_space = observation_space
        self.action_space = action_space
        self.LEARNING_RATE = LEARNING_RATE
        self.input_pl, self.gradients_pl = self.placeholder_inputs()
        self.network = NeuralNet('Actor', self.observation_space, self.action_space, NET_SIZE, TAU)
        self.prediction = self.network.inference_actor(self.input_pl, action_scale)
        self.target_network = NeuralNet('Target_Actor', self.observation_space, self.action_space, NET_SIZE, TAU)
        self.target_prediction = self.target_network.inference_actor(self.input_pl, action_scale)
        variables = self.network.get_variables()
        self.gradients = self.gradients(self.prediction, variables, self.gradients_pl)
        self.optimizer = self.optimizer(self.gradients, self.network.get_variables())

    def placeholder_inputs(self):
        """
         Generate placeholder variables to represent the input tensors
        Returns:
         input_placeholder: the tensor of the input
         labels_placeholder: the tensor for the real observations
         actions_placeholder: the tensor of output actions
        """
        input_placeholder = tf.placeholder(tf.float32,
            [None, self.observation_space], name='Input_PL')
        gradients_placeholder = tf.placeholder(tf.float32,
            [None,self.action_space], name='Gradients_PL')
        return input_placeholder, gradients_placeholder

    def gradients(self, prediction, variables, gradients_pl):
        with tf.name_scope('Actor_Gradients'):
            grads = tf.gradients(prediction, variables, -gradients_pl)
            return grads

    def optimizer(self, gradients, variables):
        with tf.name_scope('Actor_Training'):
            grads = list(zip(gradients, variables))
            opt = tf.train.AdamOptimizer(self.LEARNING_RATE)
            training_function = opt.apply_gradients(grads)
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

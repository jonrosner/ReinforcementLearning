import tensorflow as tf
import numpy as np
from NeuralNet import NeuralNet

class Critic:
    def __init__(self, sess, observation_space, action_space, LEARNING_RATE, NET_SIZE, TAU):
        self.sess = sess
        self.observation_space = observation_space
        self.action_space = action_space
        self.LEARNING_RATE = LEARNING_RATE
        self.input_pl, self.actions_pl, self.labels_pl = self.placeholder_inputs()
        self.q_net = NeuralNet("Critic", self.observation_space, self.action_space, NET_SIZE, TAU)
        self.prediction = self.q_net.inference_critic(self.input_pl, self.actions_pl)
        self.target_q_net = NeuralNet("Target_Critic", self.observation_space, self.action_space, NET_SIZE, TAU)
        self.target_prediction = self.target_q_net.inference_critic(self.input_pl, self.actions_pl)
        with tf.name_scope('Gradients'):
            self.gradients = tf.gradients(self.prediction, self.actions_pl)
        self.loss = self.loss(self.prediction, self.labels_pl)
        self.optimizer = self.optimizer(self.loss)

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
        actions_placeholder = tf.placeholder(tf.float32,
            [None, self.action_space], name='Actions_PL')
        labels_placeholder = tf.placeholder(tf.float32,
            [None, 1], name='Labels_PL')
        return input_placeholder, actions_placeholder, labels_placeholder

    def loss(self, q_vals, labels):
        """
         A function that computes the loss between q values (our predictions)
         and the labels (true values)
        Args:
         logits: the predicted reward for a certain action
         labels: the actual reward for a certain action
        Returns:
         loss: The MSE of labels and q_vals
        """
        with tf.name_scope('MSE-Loss'):
            loss = tf.reduce_mean(tf.squared_difference(labels, q_vals))
            tf.summary.scalar('Critic-Loss', loss)
            return loss

    def optimizer(self, loss):
        """
         Return a method for training the network and also update the global
         step counter
        Args:
         loss: the loss function for training
        Returns:
         training_function: a function for training the network
        """
        with tf.name_scope('Training'):
            opt = tf.train.AdamOptimizer(self.LEARNING_RATE)
            training_function = opt.minimize(loss)
            return training_function

    #def gradients(self, action_pl):
    #    with tf.name_scope('Gradients'):
    #        return tf.gradients(self.prediction, action_pl)

    def train(self, observations, actions, labels):
        """
         Train the network with a list of observations, labels and actions
        Args:
         observations: a list of past observations
         labels: the ground truth that the network tries to predict
         actions: the actions taken after each observation
        """
        feed_dict = self.fill_feed_dict(observations, actions, labels)
        _, loss, prediction = self.sess.run([self.optimizer, self.loss, self.prediction], feed_dict=feed_dict)

    def fill_feed_dict(self, observations, actions, labels):
        """
         Fills the feed_dict that will be fed to the placeholders
        Args:
         observations: the current state
        Returns:
         feed_dict: the filled feed dict
        """
        input_feed = observations
        labels_feed = labels
        actions_feed = actions
        feed_dict = {
            self.input_pl: input_feed,
            self.labels_pl: labels_feed,
            self.actions_pl: actions_feed
        }
        return feed_dict

    def get_gradients(self, observations, actions):
        labels = np.zeros(shape=[1,1])
        feed_dict = self.fill_feed_dict(observations, actions, labels)
        gradients = self.sess.run(self.gradients, feed_dict=feed_dict)
        return gradients[0]

    def predict(self, observation, action, predictor):
        """
         Predict the value of a state/action pair
        Args:
         observation: the current state
         network: the network that does the prediction (used for dual DQN)
        Returns:
         prediction: the predicted future reward
        """
        label = np.zeros(shape=[1,1])
        feed_dict = self.fill_feed_dict(observation, action, label)
        prediction = self.sess.run(predictor,feed_dict=feed_dict)
        return prediction

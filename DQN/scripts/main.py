#!/usr/bin/python

import tensorflow as tf
import numpy as np
import sys
import collections
import gym
import random
from SumTree import SumTree
from logger import Logger
import argparse

#-----------------------------------CONSTANTS-----------------------------------

NUM_INPUT = 4
NUM_HIDDEN1 = 20
NUM_HIDDEN2 = 20
NUM_ACTIONS = 2
LEARNING_RATE = 0.00025
MEMORY_LEN = 10000
MEMORY_BIAS = .01
MEMORY_POW = .6
BATCH_SIZE = 10
RND_DECREASE = .05
RND_MIN = .1
REWARD_DISCOUNT = 0.9
MAX_EPISODES = 1500
MAX_TIMESTEPS = 5000
TARGET_UPDATE_FREQ = 50 #this is in episodes, not timesteps

#-----------------------------------VARIABLES-----------------------------------

rnd_action_prob = 1
current_episode = 0

#--------------------------------Deep Q Learning--------------------------------
"""
 The Deep Neural Network tries to predict the best possible actions in an
 environment for each state it is in
"""
class DNN:
    def __init__(self, name):
        self.name = name

    def inference(self, input_pl):
        """
         Create the network that will be used to predict q values
        Args:
         input_pl: a placeholder for the inputs of the network
        """
        global NUM_INPUT, NUM_ACTIONS, NUM_HIDDEN1, NUM_HIDDEN2
        with tf.variable_scope(self.name):
            with tf.name_scope("Hidden1"):
                with tf.name_scope("Weights"):
                    W1 = tf.Variable(tf.truncated_normal(
                    [NUM_INPUT, NUM_HIDDEN1], stddev=1.0), name='Weights')
                with tf.name_scope("Biases"):
                    b1 = tf.Variable(tf.zeros([NUM_HIDDEN1]), name='Biases')
                h1 = tf.tanh(tf.add(tf.matmul(input_pl, W1), b1))

#----------------------------------DUELING DQN----------------------------------

            with tf.name_scope("Value"):
                with tf.name_scope("Weights"):
                    W_V = tf.Variable(tf.truncated_normal([NUM_HIDDEN1, 1],
                    stddev=1.0), name='Weights')
                with tf.name_scope("Biases"):
                    b_V = tf.Variable(tf.zeros([1]), name='Biases')
                value = tf.add(tf.matmul(h1, W_V), b_V)

            with tf.name_scope("Advantage"):
                with tf.name_scope("Weights"):
                    W_A = tf.Variable(tf.truncated_normal(
                    [NUM_HIDDEN1, NUM_ACTIONS], stddev=1.0), name='Weights')
                with tf.name_scope("Biases"):
                    b_A = tf.Variable(tf.zeros([NUM_ACTIONS]), name='Biases')
                advantage = tf.add(tf.matmul(h1, W_A), b_A)

#-------------------------------------------------------------------------------
            with tf.name_scope("Output"):
                out = tf.add(value, tf.subtract(advantage,
                tf.reduce_mean(advantage, axis=1, keep_dims=True)))
                tf.summary.histogram('Output', out)
            return out

    def variable_summaries(self, var):
        """
         Attaches summaries to a tensor for visualization in tensorboard
        Args:
         var: the tensor to be visualized
        """
        with tf.name_scope('summaries'):
            mean = tf.reduce_mean(var)
            tf.summary.scalar('mean', mean)
            with tf.name_scope('stddev'):
                stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
                tf.summary.scalar('stddev', stddev)
            tf.summary.scalar('max', tf.reduce_max(var))
            tf.summary.scalar('min', tf.reduce_min(var))
            tf.summary.histogram('histogram', var)

    def get_variables(self):
        """
         Gets all weights and biases of the network
        Returns:
         A collection of all trainable variables
        """
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.name)

    def copy_to(self, target_network, sess):
        """
         Copies all trainable variables of this network into another network of
         same size
        Args:
         target_network: the network to be copied into
         sess: the tensorflow session that runs the assign-operations
        """
        v1 = self.get_variables()
        v2 = target_network.get_variables()
        for i in range(len(v1)):
            v2[i].assign(v1[i]).eval(session=sess)

"""
 The actor that will learn how to optimally behave in a certain environment
"""
class QLearning:
    def __init__(self):
        self.q_net = DNN("QNet")
        #Double DQN
        self.target_q_net = DNN("TargetQNet")
        self.replay_memory = Replay_Memory()
        self.input_pl, self.labels_pl, self.actions_pl=self.placeholder_inputs()
        outputs = self.q_net.inference(self.input_pl)
        self.predictions = outputs
        target_outputs = self.target_q_net.inference(self.input_pl)
        self.target_predictions = target_outputs
        self.q_vals = self.q_vals(self.predictions, self.actions_pl)
        self.loss = self.loss(self.q_vals, self.labels_pl)
        self.train_op = self.training(self.loss)
        #this tensor is to log the received rewards
        self.reward = 0.0
        self.reward_pl = tf.placeholder(tf.float32, name='Reward_pl')
        self.reward_f = tf.add(0.0, self.reward_pl)
        self.q_net.variable_summaries(self.reward_f)
        tf.summary.scalar('reward', self.reward_f)
        init = tf.global_variables_initializer()
        self.sess = tf.Session()
        #this is for logging the results in tensorboard
        self.writer = tf.summary.FileWriter("logs/", self.sess.graph)
        self.merged = tf.summary.merge_all()
        self.sess.run(init)


    def placeholder_inputs(self):
        """
         Generate placeholder variables to represent the input tensors
        Returns:
         input_placeholder: the tensor of the input
         labels_placeholder: the tensor for the real observations
         actions_placeholder: the tensor of output actions
        """
        global NUM_ACTIONS, NUM_INPUT
        input_placeholder = tf.placeholder(tf.float32,
            [None, NUM_INPUT], name='Input_PL')
        labels_placeholder = tf.placeholder(tf.float32,
            [None,], name='Labels_PL')
        actions_placeholder = tf.placeholder(tf.float32,
            [None, NUM_ACTIONS], name='Actions_PL')
        return input_placeholder, labels_placeholder, actions_placeholder

    def q_vals(self, predictions, actions_pl):
        """
         The q values to are needed for our loss function
        Returns:
         q_vals: a function to compute q values
        """
        global BATCH_SIZE
        with tf.name_scope('Q-Values'):
            q_vals = tf.reduce_sum(tf.multiply(predictions, actions_pl), 1)
            return q_vals

    def loss(self, q_vals, labels):
        """
         A function that computes the loss between q values (our predictions)
         and the labels (true values)
        Args:
         logits: the predicted reward for a certain action
         labels: the actual reward for a certain action
        Returns:
         loss_function: a function that when run computes the loss between
                        logits and labels
        """
        with tf.name_scope('MSE-Loss'):
            loss = tf.reduce_sum(tf.square(labels - q_vals))
            tf.summary.scalar('loss', loss)
            return loss

    def training(self, loss):
        """
         Return a method for training the network and also update the global
         step counter
        Args:
         loss: the loss function for training
        Returns:
         training_function: a function for training the network
        """
        global LEARNING_RATE
        with tf.name_scope('Training'):
            optimizer = tf.train.GradientDescentOptimizer(LEARNING_RATE)
            training_function = optimizer.minimize(loss)
            return training_function

    def fill_feed_dict(self, observations, labels, actions):
        """
         Fills the feed_dict that will be fed to the placeholders
        Args:
         observations: the current state of the pendulum
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

    def train_step(self, observations, labels, actions, summary):
        """
         Train the network with a list of observations, labels and actions
        Args:
         observations: a list of past observations
         labels: the ground truth that the network tries to predict
         actions: the actions taken after each observation
        """
        global current_episode
        feed_dict = self.fill_feed_dict(observations, labels, actions)
        if not summary:
            loss, _, predictions, q_vals = self.sess.run([self.loss,
                self.train_op, self.predictions, self.q_vals],
                feed_dict=feed_dict)
        else:
            feed_dict[self.reward_pl] = self.reward
            summary = self.sess.run(self.merged, feed_dict)
            self.writer.add_summary(summary, current_episode)


    def predict(self, observation, network):
        """
         Tries to predict the value of each possible action in a state
        Args:
         observation: the current state
         network: the network that does the prediction (used for dueling DQN)
        Returns:
         prediction: the predicted future reward of all actions
        """
        global NUM_ACTIONS
        action = np.zeros((len(observation), NUM_ACTIONS))
        label = np.zeros(len(observation))
        feed_dict = self.fill_feed_dict(observation, label, action)
        prediction = self.sess.run(network,feed_dict=feed_dict)
        return prediction

    def save_transition(self, observation, action,
        new_observation, reward, done):
        """
         Add the given transition into a replay memory for later training.
        Args:
         observation: the past state
         action: the action taken
         new_observation: the new state
         reward: the reward received
         done: if the episode is to be continued or not
        """
        global REWARD_DISCOUNT
        transition = {
            'observation': observation,
            'action': action,
            'new_observation': new_observation,
            'reward': reward,
            'done': done
        }
        error = self.calcError(observation, new_observation, reward, action)
        self.replay_memory.add(error, transition)

    def select_action(self, observation):
        """
         Select best action in a state due to prediction or choose a random one
        Args:
         observation: the current state of the pendulum
        Returns:
         action: the action to be taken next
        """
        global rnd_action_prob
        if random.random() < rnd_action_prob:
            action = np.random.randint(0, NUM_ACTIONS)
        else:
            predictions = self.predict(observation, self.predictions)
            action = np.argmax(predictions)
        return action

    def train_with_batch(self, summary):
        """
         Call train_step with a sample batch from the replay memory
        Args:
         summary: boolean if the training results are to be saved in a logfile
        """
        global NUM_ACTIONS, REWARD_DISCOUNT, BATCH_SIZE, NUM_INPUT, \
            current_episode,TARGET_UPDATE_FREQ

        if summary and current_episode % TARGET_UPDATE_FREQ == 0:
            self.q_net.copy_to(self.target_q_net, self.sess)

        batch = self.replay_memory.sample()

        observations = np.zeros((len(batch), NUM_INPUT))
        labels = np.zeros(len(batch))
        errors = np.zeros(len(batch))
        actions = []
        i = 0
        for idx, sample in batch:
            observations[i] = sample['observation']
            if not sample['done']:
                new_observation = sample['new_observation']
                argmax_q = np.argmax(self.predict(new_observation, self.predictions))
                #advantage would be adv = q_vals - np.mean(q_vals) / len(q_vals)
                q_vals = self.predict(new_observation, self.target_predictions)[0]
                labels[i] = sample['reward'] + REWARD_DISCOUNT * q_vals[argmax_q]
                error = self.calcError(np.array([observations[i]]),
                    new_observation, sample['reward'], sample['action'])
                self.replay_memory.update(idx, error)
            else:
                labels[i] = sample['reward']
            action = np.zeros(NUM_ACTIONS)
            action[sample['action']] = 1
            actions.append(action)
            i += 1
        self.train_step(observations, labels, actions, summary)

    def calcError(self, observation, new_observation, reward, action):
        """
         Calculates the error that determines the usefullness of a memory.
         High errors are better for training
        Args:
         observation: the old state
         new_observation: the current state
         reward: the reward received
         action: the action that was taken
        Returns:
         error: the difference between prediction and label
        """
        prediction = self.predict(observation, self.predictions)[0][action]
        label = reward + REWARD_DISCOUNT * self.predict(new_observation,
            self.target_predictions)[0][action]
        error = abs(label - prediction)
        return error

#-------------------------PRIORITIZED EXPERIENCE REPLAY-------------------------

#from: https://github.com/jaara/AI-blog/blob/master/Seaquest-DDQN-PER.py
#credits to: Jaromir Janisch
"""
The Memory that is used for Prioritized Experience Replay
"""
class Replay_Memory:
    def __init__(self):
        global MEMORY_LEN
        self.tree = SumTree(MEMORY_LEN)

    def add(self, error, sample):
        global MEMORY_BIAS, MEMORY_POW
        priority = (error + MEMORY_BIAS) ** MEMORY_POW
        self.tree.add(priority, sample)

    def sample(self):
        """
         Get a sample batch of the replay memory
        Returns:
         batch: a batch with one sample from each segment of the memory
        """
        global BATCH_SIZE
        batch = []
        #we want one representative of all distribution-segments in the batch
        #e.g BATCH_SIZE=2: batch contains one sample from [min,median]
        #and from [median,max]
        segment = self.tree.total() / BATCH_SIZE
        for i in range(BATCH_SIZE):
            minimum = segment * i
            maximum = segment * (i+1)
            s = random.uniform(minimum, maximum)
            (idx, p, data) = self.tree.get(s)
            batch.append((idx, data))
        return batch

    def update(self, idx, error):
        """
         Updates one entry in the replay memory
        Args:
         idx: the position of the outdated transition in the memory
         error: the newly calculated error
        """
        priority = (error + MEMORY_BIAS) ** MEMORY_POW
        self.tree.update(idx, priority)

#--------------------------------------MAIN-------------------------------------

"""
The Main Loop of the algorithm
"""
def run(_):
    global current_episode, rnd_action_prob, RND_MIN, RND_DECREASE
    env = gym.make(FLAGS.environment)
    #env = gym.wrappers.Monitor(env, '/tmp/cartpole-experiment-1', force=True)
    dqn = QLearning()
    logger = Logger(env, dqn.sess, FLAGS)
    last_100 = collections.deque(maxlen=100)

    for episode in range(MAX_EPISODES):
        observation = env.reset()
        observation = np.array([observation])
        episode_reward = 0.0
        current_episode += 1
        for t in range(MAX_TIMESTEPS):
            #env.render()
            action = dqn.select_action(observation)
            new_observation, reward, done, info = env.step(action)
            episode_reward += reward
            new_observation = np.array([new_observation])
            dqn.save_transition(observation, action,
                new_observation, reward, done)
            observation = new_observation
            dqn.train_with_batch(summary=False)
            if done:
                print("Episode ", episode)
                print("Finished after %s timesteps." % t)
                print("Reward this episode: %s" % episode_reward)
                last_100.append(episode_reward)
                logger.add_reward(episode_reward)
                print("Avg Reward last 100 episodes: ", np.mean(last_100))
                print("RND Action: ", rnd_action_prob)
                break
        #decrease random probability
        if rnd_action_prob >= RND_MIN: rnd_action_prob -= RND_DECREASE
        if np.mean(last_100) >= 195:
                gym.upload('/tmp/cartpole-experiment-1',
                    api_key=sys.argv[1])
                exit()
        dqn.reward = episode_reward
        dqn.train_with_batch(summary=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
      '--environment',
      type=str,
      default='CartPole-v0',
      help='The OpenAI-gym environment name.'
    )
    parser.add_argument(
      '--log_dir',
      type=str,
      default='../logs',
      help='Directory to put the log data.'
    )
    parser.add_argument(
      '--verbose',
      default=False,
      help='Prints helpfull debugging information every 500 timesteps.',
      action='store_true'
    )
    parser.add_argument(
      '--gym_api_key',
      type=str,
      help='Your own API-Key to upload your results to your OpenAI-Gym profile.'
    )
    parser.add_argument(
      '--gym_upload_threshhold',
      type=float,
      default=999999.,
      help='Upload your results when this reward-threshhold is reached. \
        Does not upload if not set. Will cancel learning.',
    )
    parser.add_argument(
      '--load_trained_model_path',
      type=str,
      default='',
      help='Loads a trained model from the given path. Does not load if not set.'
    )

    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=run, argv=[sys.argv[0]] + unparsed)

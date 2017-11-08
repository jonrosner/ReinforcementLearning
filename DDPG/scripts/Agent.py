import tensorflow as tf
import numpy as np
from Actor import Actor
from Critic import Critic
from ReplayMemory import ReplayMemory
import random

class Agent:
    def __init__(self, env, sess, LEARNING_RATE_ACTOR, LEARNING_RATE_CRITIC,
            NET_SIZE, MEMORY_LEN, REWARD_DISCOUNT, BATCH_SIZE, TAU,
            EXPLORATION_STEPS, VERBOSE, LOG_DIR_TF):
        self.env = env
        self.sess = sess
        self.observation_space = self.env.observation_space.shape[0]
        self.action_space = self.env.action_space.shape[0]
        self.REWARD_DISCOUNT = REWARD_DISCOUNT
        self.TAU = TAU
        self.BATCH_SIZE = BATCH_SIZE
        self.noise_state = np.zeros(self.action_space)
        self.EXPLORATION_STEPS = EXPLORATION_STEPS
        self.VERBOSE = VERBOSE
        self.LOG_DIR_TF = LOG_DIR_TF
        #check if action_space is symmetric
        if all(env.action_space.high == abs(env.action_space.low)):
            action_scale = env.action_space.high
        else:
            raise ActionSpaceNotSymmetricException
        self.actor = Actor(self.sess,self. observation_space, self.action_space, LEARNING_RATE_ACTOR, NET_SIZE, TAU, action_scale)
        self.critic = Critic(self.sess, self.observation_space, self.action_space, LEARNING_RATE_CRITIC, NET_SIZE, TAU)
        actor_network_variables = self.actor.network.get_variables()
        critic_q_net_variables = self.critic.q_net.get_variables()
        self.actor_target_update = self.actor.target_network.update_variables(actor_network_variables)
        self.critic_target_update = self.critic.target_q_net.update_variables(critic_q_net_variables)
        self.reward_pl = tf.placeholder(tf.float32, [None, 1], name='Reward_PL')
        self.done_pl = tf.placeholder(tf.bool, [None, 1], name='Done_PL')
        self.labels = tf.where(self.done_pl, self.reward_pl, self.reward_pl + tf.multiply(self.REWARD_DISCOUNT, self.critic.target_prediction))
        #self.replay_memory = ReplayMemory(MEMORY_LEN, BATCH_SIZE)
        self.replay_memory = ReplayMemory(MEMORY_LEN, BATCH_SIZE, self.observation_space, self.action_space)
        self.log_reward_pl = tf.placeholder(tf.float32, name='Reward_log_pl')
        self.reward_f = tf.add(0.0, self.log_reward_pl)
        tf.summary.scalar('reward', self.reward_f)
        init = tf.global_variables_initializer()
        self.sess.run(init)
        self.sess.run(self.actor.network.copy_to(self.actor.target_network))
        self.sess.run(self.critic.q_net.copy_to(self.critic.target_q_net))
        self.writer = tf.summary.FileWriter(self.LOG_DIR_TF, self.sess.graph)
        self.merged = tf.summary.merge_all()

    def select_action(self, observation, current_step):
        action = self.actor.predict(observation, self.actor.prediction)
        if current_step <= self.EXPLORATION_STEPS:
            noise = self.noise()
        else:
            noise = 0
        return action + noise

    def noise(self):
        x = self.noise_state
        dx = 0.15 * (0 - x) + 0.2 * np.random.randn(len(x))
        self.noise_state = x + dx
        return self.noise_state

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
        prediction = self.critic.predict(observation, action, self.critic.prediction)
        label = reward + self.REWARD_DISCOUNT * self.critic.predict(new_observation, action,
            self.critic.target_prediction)
        error = abs(label - prediction)
        return error

    def summarize(self, episode, episode_reward, observation, new_observation, reward, done):
        next_action = self.actor.predict(new_observation, self.actor.target_prediction)
        feed_dict = {
            self.critic.input_pl: new_observation,
            self.critic.actions_pl: next_action,
            self.reward_pl: [[reward]],
            self.done_pl: [[done]]
        }
        label = self.sess.run(self.labels, feed_dict=feed_dict)
        feed_dict[self.critic.labels_pl] = label
        #sometimes the reward is an array and sometimes a scalar
        if isinstance(episode_reward, np.ndarray):
            episode_reward = max(episode_reward)
        feed_dict[self.log_reward_pl] = episode_reward
        summary = self.sess.run(self.merged, feed_dict=feed_dict)
        self.writer.add_summary(summary, episode)

    def train_with_batch(self, current_step):
        """
         Call train_step with a sample batch from the replay memory
        Args:
         summary: boolean if the training results are to be saved in a logfile
        """
        observations, actions, rewards, new_observations, dones = self.replay_memory.sample()
        #all of this requires ~3 seconds of computational time
        #improve the Q-Network
        next_actions = self.actor.predict(new_observations, self.actor.prediction)
        feed_dict = {
            self.critic.input_pl: new_observations,
            self.critic.actions_pl: next_actions,
            self.reward_pl: rewards,
            self.done_pl: dones
        }
        labels = self.sess.run(self.labels, feed_dict=feed_dict)
        self.critic.train(observations, actions, labels)
        actions = self.actor.predict(observations, self.actor.prediction)
        gradients = self.critic.get_gradients(observations, actions)
        #improve the policy with the calculated gradients
        self.actor.train(observations, gradients)
        #Update both target networks
        #requires ~1 second of time
        self.sess.run(self.actor_target_update)
        self.sess.run(self.critic_target_update)
        #Print debug information if verbose
        if current_step % 500 == 0 and self.VERBOSE:
            print("Observations: ", observations)
            print("Predicted Best-Actions: ", actions)
            print("Labels: ", labels)
            print("Gradients: ", gradients)

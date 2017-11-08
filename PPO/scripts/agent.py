import tensorflow as tf
import numpy as np
import gym
from neural_net import Network
from logger import Logger
from memory import ReplayMemory
from memory import BatchGenerator
from policy import Policy
from value_function import Value_function
import scipy.signal

class Agent:
    def __init__(self, env, sess, horizon, epsilon, learning_rate_policy, learning_rate_value, gamma, lam, logger):
        self.env = env
        self.sess = sess
        self.horizon = horizon
        self.epsilon = epsilon
        self.learning_rate_policy = learning_rate_policy
        self.learning_rate_value = learning_rate_value
        self.gamma = gamma
        self.lam = lam
        self.logger = logger

        self.observation_space = env.observation_space.shape[0]
        self.action_space = env.action_space.shape[0]

        self.policy = Policy(self.observation_space, self.action_space, self.epsilon, self.learning_rate_policy)
        self.value_function = Value_function(self.observation_space, self.learning_rate_value)

        self.replay_memory = ReplayMemory(self.horizon, self.observation_space, self.action_space)

    def learn(self):
        """
         Learning process that loops forever if not stopped
        """
        while True:
            #Fill replay memory with one trajectory
            self.run_trajectory()
            adv, vtarget = self.gae()
            self.sess.run(self.policy.network.copy_to(self.policy.network_old))
            #Train policy and value function on minibatch
            bg = BatchGenerator((self.replay_memory.observations, self.replay_memory.actions, adv), 1000)
            for _ in range(20):
                for ms, ma, madv in bg.iterate_once():
                    self.sess.run(self.policy.optimizer, {self.policy.network.input_pl: ms, self.policy.network_old.input_pl: ms, self.policy.action_pl: ma, self.policy.adv_pl: madv})
            bg = BatchGenerator((self.replay_memory.observations, vtarget), 250)
            for _ in range(10):
                for ms, mvpred in bg.iterate_once():
                    self.sess.run(self.value_function.optimizer, {self.value_function.network.input_pl: ms, self.value_function.value_pl: mvpred})


    def run_trajectory(self):
        """
         Runs for one trajectory and fills the replay memory
        Returns:
         Nothing, data is stored in replay memory for later use
        """
        self.replay_memory.clear()
        observation = self.env.reset()
        episode_reward = 0
        for _ in range(self.horizon):
            observation = np.array([observation])
            action = self.sess.run(self.policy.network.sample, {self.policy.network.input_pl: observation})[0]
            new_observation, reward, done, info = self.env.step(action)
            episode_reward += reward
            self.replay_memory.add(observation, action, reward, new_observation, done)
            if done:
                #Log episode reward and reset
                self.logger.add_reward(episode_reward)
                episode_reward = 0
                observation = self.env.reset()
            else:
                observation = new_observation

    def gae(self):
        """
         Takes data in replay memory and calculates general advantage estimate with it
        Returns:
         gae: general advantage estimate
         vtarget: predicted values
        """
        v = self.sess.run(self.value_function.network.predict, {self.value_function.network.input_pl: self.replay_memory.observations})
        v1 = self.sess.run(self.value_function.network.predict, {self.value_function.network.input_pl: self.replay_memory.new_observations})
        tds = self.replay_memory.rewards + self.gamma * v1 * (1 - self.replay_memory.done) - v
        gae = scipy.signal.lfilter([1.0], [1.0, -self.gamma * self.lam], tds[::-1])[::-1]
        vtarget = gae + v
        gae = (gae - gae.mean()) / (gae.std() + 1e-6)
        return gae, vtarget

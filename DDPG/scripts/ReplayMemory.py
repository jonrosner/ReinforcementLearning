import random
import numpy as np

class ReplayMemory:
    def __init__(self, MEMORY_LEN, BATCH_SIZE, observation_space, action_space):
        self.MEMORY_LEN = MEMORY_LEN
        self.BATCH_SIZE = BATCH_SIZE
        self.observation_space = observation_space
        self.action_space = action_space
        #Init Empty Memory
        self.counter = 0
        self.current_len = 0
        self.observations = np.empty([MEMORY_LEN, self.observation_space], dtype=np.float32)
        self.actions = np.empty([MEMORY_LEN, self.action_space], dtype=np.float32)
        self.rewards = np.empty([MEMORY_LEN, 1], dtype=np.float32)
        self.new_observations = np.empty([MEMORY_LEN, self.observation_space], dtype=np.float32)
        self.done = np.empty([MEMORY_LEN, 1], dtype=np.bool)

    def add(self, observation, action, reward, new_observation, done):
        self.observations[self.counter] = observation
        self.actions[self.counter] = action
        self.rewards[self.counter] = reward
        self.new_observations[self.counter] = new_observation
        self.done[self.counter] = done
        if self.counter < self.MEMORY_LEN-1:
            self.counter += 1
        else:
            self.counter = 0
        if self.current_len != self.MEMORY_LEN:
            self.current_len += 1

    def sample(self):
        """
         Get a sample batch of the replay memory
        Returns:
         batch: a batch with one sample from each segment of the memory
        """
        idxs = np.random.randint(0, self.current_len, self.BATCH_SIZE)
        observation_batch = self.observations[idxs]
        action_batch = self.actions[idxs]
        reward_batch = self.rewards[idxs]
        new_observation_batch = self.new_observations[idxs]
        done_batch = self.done[idxs]
        return observation_batch, action_batch, reward_batch, new_observation_batch, done_batch

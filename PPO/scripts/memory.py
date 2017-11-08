import numpy as np

class ReplayMemory:
    def __init__(self, memory_len, observation_space, action_space):
        self.memory_len = memory_len
        self.observation_space = observation_space
        self.action_space = action_space

    def add(self, observation, action, reward, new_observation, done):
        """
         Add one timestep to the replay memory
        Args:
         observation: previous state
         action: action taken
         reward: observered reward
         new_observation: new state
         done: is the episode finished?
        """
        self.observations[self.counter] = observation
        self.actions[self.counter] = action
        self.rewards[self.counter] = reward
        self.new_observations[self.counter] = new_observation
        self.done[self.counter] = done
        if self.counter < self.memory_len-1:
            self.counter += 1

    def clear(self):
        """
         Clears the replay memory and resets counter
        """
        self.counter = 0
        self.observations = np.empty([self.memory_len, self.observation_space], dtype=np.float32)
        self.actions = np.empty([self.memory_len, self.action_space], dtype=np.float32)
        self.rewards = np.empty([self.memory_len], dtype=np.float32)
        self.new_observations = np.empty([self.memory_len, self.observation_space], dtype=np.float32)
        self.done = np.empty([self.memory_len], dtype=np.bool)


"""
 Creates a minibatch used for training.
 Thanks to Julius F. for this piece of code.
"""
class BatchGenerator:
    def __init__(self, data, minibatch_size):
        self.data = data
        self.minibatch_size = minibatch_size

    def iterate_once(self):
        per = np.random.permutation(np.arange(len(self.data[0])))
        if len(self.data[0]) % self.minibatch_size != 0:
            np.append(per, np.random.random_integers(0, len(self.data[0]), len(self.data[0]) % self.minibatch_size))
        for i in range(0, len(per), self.minibatch_size):
            p = per[i:i + self.minibatch_size]
            r = [None] * len(self.data)
            for j in range(len(self.data)):
                r[j] = self.data[j][p]
            yield tuple(r)

import os
import tensorflow as tf
import datetime
import json
import csv
import time
import gym
import collections

class Logger:
    def __init__(self, env, sess, FLAGS):
        self.env = env
        self.sess = sess
        #self.saver = tf.train.Saver(max_to_keep=10, keep_checkpoint_every_n_hours=1)
        time_str = datetime.datetime.now().strftime("%b-%d_%H-%M-%S")
        self.log_dir_tf = os.path.join(FLAGS.log_dir, FLAGS.environment, time_str, 'tf')
        log_dir_gym = os.path.join(FLAGS.log_dir, FLAGS.environment, time_str, 'gym')
        arguments_json_path = os.path.join(FLAGS.log_dir, FLAGS.environment, time_str, 'arguments.json')
        self.results_csv_path = os.path.join(FLAGS.log_dir, FLAGS.environment, time_str, 'rewards.csv')
        self.env = gym.wrappers.Monitor(self.env, log_dir_gym, force=True)
        #Delete and recreate tf's log dir if it already exists
        if tf.gfile.Exists(self.log_dir_tf):
            tf.gfile.DeleteRecursively(self.log_dir_tf)
        tf.gfile.MakeDirs(self.log_dir_tf)
        #Dump flags
        with open(arguments_json_path, 'w') as outfile:
            print("Dumping arguments to Json file: ", arguments_json_path)
            json.dump(vars(FLAGS), outfile)
        self.last_100 = collections.deque(maxlen=100)
        self.current_step = 0
        self.current_episode = 0

    #Adds rewards to the loggers csv file
    def add_reward(self, reward):
        """
         Writes the reward to file.
        Args:
         reward: numerical reward to be saved
        """
        with open(self.results_csv_path, 'a') as outfile:
            writer = csv.writer(outfile, delimiter=',')
            #sometimes the reward is an array and sometimes scalar
            if isinstance(reward, float):
                reward = [reward]
            writer.writerow(reward)
        print("Reward this episode: ", reward)

import gym
import tensorflow as tf
import datetime
from agent import Agent
from logger import Logger
import argparse
import sys

def run(_):
    env = gym.make(FLAGS.environment)
    sess = tf.Session()
    logger = Logger(env, sess, FLAGS)
    agent = Agent(env,
            sess,
            FLAGS.horizon,
            FLAGS.epsilon,
            FLAGS.learning_rate_policy,
            FLAGS.learning_rate_value,
            FLAGS.gamma,
            FLAGS.lam,
            logger)
    sess.run(tf.global_variables_initializer())
    
    #Start learning process
    agent.learn()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
      '--environment',
      type=str,
      default='Swimmer-v1',
      help='The OpenAI-gym environment name.'
    )
    parser.add_argument(
      '--horizon',
      type=int,
      default=5000,
      help='The number of timesteps in a trajectory.'
    )
    parser.add_argument(
      '--learning_rate_policy',
      type=float,
      default=0.00025,
      help='Learning rate of policy.'
    )
    parser.add_argument(
      '--learning_rate_value',
      type=float,
      default=0.0025,
      help='Learning rate of value function.'
    )
    parser.add_argument(
      '--epsilon',
      type=float,
      default=0.2,
      help='Surrogate clipping value.'
    )
    parser.add_argument(
      '--gamma',
      type=float,
      default=1.,
      help='Discount factor.'
    )
    parser.add_argument(
      '--lam',
      type=float,
      default=0.98,
      help='GAE parameter.'
    )
    parser.add_argument(
      '--log_dir',
      type=str,
      default='../logs',
      help='Directory to put the log data.'
    )
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=run, argv=[sys.argv[0]] + unparsed)

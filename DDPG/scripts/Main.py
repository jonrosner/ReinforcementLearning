import tensorflow as tf
import numpy as np
import gym
from Agent import Agent
import argparse
import sys
import collections
import datetime
import json
import csv
import time

def run(_):
    time_str = datetime.datetime.now().strftime("%d-%m-%Y-%M-%S")
    sess = tf.Session()
    #Define the logging folder structure
    log_dir_tf = FLAGS.log_dir + '/' + FLAGS.environment + '/' + time_str + '/tf/'
    log_dir_gym = FLAGS.log_dir + '/' + FLAGS.environment + '/' + time_str + '/gym/'
    arguments_json_path = FLAGS.log_dir + '/' + FLAGS.environment + '/' + time_str + '/' + 'arguments.json'
    results_csv_path = FLAGS.log_dir + '/' + FLAGS.environment + '/' + time_str + '/' + 'rewards.csv'
    #Delete and recreate tf's log dir if it already exists
    if tf.gfile.Exists(log_dir_tf):
        tf.gfile.DeleteRecursively(log_dir_tf)
    tf.gfile.MakeDirs(log_dir_tf)
    #Dump all arguments to json file
    with open(arguments_json_path, 'w') as outfile:
        print("Dumping arguments to Json file: ", arguments_json_path)
        json.dump(vars(FLAGS), outfile)
    env = gym.make(FLAGS.environment)
    env = gym.wrappers.Monitor(env, log_dir_gym, force=True)
    NET_SIZE = [FLAGS.hidden1, FLAGS.hidden2]
    current_step = 0
    last_100 = collections.deque(maxlen=100)
    agent = Agent(env,
                sess,
                FLAGS.learning_rate_actor,
                FLAGS.learning_rate_critic,
                NET_SIZE,
                FLAGS.memory_len,
                FLAGS.reward_discount,
                FLAGS.batch_size,
                FLAGS.tau,
                FLAGS.exploration_steps,
                FLAGS.verbose,
                log_dir_tf)
    saver = tf.train.Saver(max_to_keep=10, keep_checkpoint_every_n_hours=12)
    #Load trained model if provided
    if FLAGS.load_trained_model_path != '':
        print("Loading trained model from ", FLAGS.load_trained_model_path)
        saver.restore(sess, FLAGS.load_trained_model_path)
    #Save initial model
    saver.save(sess, log_dir_tf + 'my_test_model', global_step=0)
    #Main training-loop
    for episode in range(FLAGS.max_episodes):
        observation = env.reset()
        observation = np.array([observation])
        episode_reward = 0.0
        start_time = time.time()
        for timestep in range(env.spec.timestep_limit):
            if current_step > 1000 and episode % 10 == 0:
                env.render()
            action = agent.select_action(observation, current_step)
            new_observation, reward, done, info = env.step(action)
            #reward = max(reward)
            episode_reward += reward
            #TODO: fix workaround
            new_observation = np.reshape(new_observation, (1, env.observation_space.shape[0]))
            agent.replay_memory.add(observation, action, reward, new_observation, done)
            old_observation = observation
            observation = new_observation
            if current_step > 10000:
                agent.train_with_batch(current_step)
            current_step += 1
            if done or timestep == env.spec.timestep_limit:
                end_time = time.time()
                print("Episode ", episode)
                print("Finished after %s timesteps. (%s seconds elapsed)" % (timestep, (end_time - start_time)))
                print("Reward this episode: %s" % episode_reward)
                last_100.append(episode_reward)
                print("Avg Reward last 100 episodes: ", np.mean(last_100))
                agent.summarize(episode, episode_reward, old_observation, new_observation, reward, done)
                with open(results_csv_path, 'a') as outfile:
                    writer = csv.writer(outfile, delimiter=',')
                    #sometimes the reward is an array and sometimes scalar
                    if isinstance(episode_reward, float):
                        episode_reward = [episode_reward]
                    writer.writerow(episode_reward)
                break
        #Update save file after every episode
        saver.save(sess, log_dir_tf + 'my_test_model' + str(current_step), write_meta_graph=False)
        if np.mean(last_100) >= FLAGS.gym_upload_threshhold and FLAGS.gym_api_key != '':
            env.close()
            gym.upload(FLAGS.log_dir_gym, api_key=FLAGS.gym_api_key)
            return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
      '--environment',
      type=str,
      default='Pendulum-v0',
      help='The OpenAI-gym environment name.'
    )
    parser.add_argument(
      '--learning_rate_actor',
      type=float,
      default=0.0001,
      help='Learning rate of the policy network.'
    )
    parser.add_argument(
      '--learning_rate_critic',
      type=float,
      default=0.001,
      help='Learning rate of the value network.'
    )
    parser.add_argument(
      '--max_episodes',
      type=int,
      default=1000000,
      help='Maximum number of training episodes.'
    )
    parser.add_argument(
      '--hidden1',
      type=int,
      default=200,
      help='Number of units in hidden layer 1.'
    )
    parser.add_argument(
      '--hidden2',
      type=int,
      default=200,
      help='Number of units in hidden layer 2.'
    )
    parser.add_argument(
      '--batch_size',
      type=int,
      default=10,
      help='The size of the sampled batch from the replay memory'
    )
    parser.add_argument(
      '--log_dir',
      type=str,
      default='../logs',
      help='Directory to put the log data.'
    )
    parser.add_argument(
      '--memory_len',
      type=int,
      default=1000000,
      help='The maximum amount of transitions stored in the replay memory.',
    )
    parser.add_argument(
      '--reward_discount',
      type=float,
      default=0.99,
      help='The value factor of future predictions.',
    )
    parser.add_argument(
      '--tau',
      type=float,
      default=0.001,
      help='The updating rate for the target networks.',
    )
    parser.add_argument(
      '--exploration_steps',
      type=int,
      default=9999999,
      help='The number of steps that exploration noise will be added to the \
            predictions.',
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

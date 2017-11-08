import tensorflow as tf

class NeuralNet:
    def __init__(self, name, observation_space, action_space, NET_SIZE, TAU):
        #TODO: target network update every timestep
        self.name = name
        self.observation_space = observation_space
        self.action_space = action_space
        self.NET_SIZE = NET_SIZE
        self.TAU = TAU

    def weight_variable(self, shape, stddev, name):
        initial = tf.truncated_normal(shape, stddev=stddev)
        return tf.Variable(initial, name=name)

    def bias_variable(self, shape, name):
        initial = tf.zeros(shape)
        return tf.Variable(initial, name=name)

    def inference_actor(self, input_pl, action_scale):
        """
         Create the network that will be used to predict q values
        Args:
         input_pl: a placeholder for the inputs of the network
        """
        with tf.variable_scope(self.name):
            with tf.name_scope("Hidden1"):
                W1 = self.weight_variable([self.observation_space, self.NET_SIZE[0]], .1, 'Weights')
                b1 = self.bias_variable(self.NET_SIZE[0], 'Biases')
                h1 = tf.nn.relu(tf.add(tf.matmul(input_pl, W1), b1))
            with tf.name_scope("Hidden2"):
                W2 = self.weight_variable([self.NET_SIZE[0], self.NET_SIZE[1]], .1, 'Weights')
                b2 = self.bias_variable(self.NET_SIZE[1], 'Biases')
                h2 = tf.nn.relu(tf.add(tf.matmul(h1, W2), b2))
            with tf.name_scope("Output"):
                W_out = self.weight_variable([self.NET_SIZE[1], self.action_space], .1, 'Weights')
                b_out = self.bias_variable(self.action_space, 'Biases')
                #TODO: Fix workaround to scale output from [-1 1] to [-2 2]
                out = tf.nn.tanh(tf.add(tf.matmul(h2, W_out), b_out))
                scaled_out = tf.multiply(out, action_scale)
                self.variables = [W1, b2, W2, b2, W_out, b_out]
                return scaled_out

    def inference_critic(self, input_obs_pl, input_action_pl):
        """
         Create the network that will be used to predict q values
        Args:
         input_pl: a placeholder for the inputs of the network
        """
        with tf.variable_scope(self.name):
            with tf.name_scope("Hidden1"):
                W1 = self.weight_variable([self.observation_space, self.NET_SIZE[0]], .1, 'Weights')
                b1 = self.bias_variable(self.NET_SIZE[0], 'Biases')
                h1 = tf.nn.relu(tf.add(tf.matmul(input_obs_pl, W1), b1))
            with tf.name_scope("Hidden2"):
                W2 = self.weight_variable([self.NET_SIZE[0], self.NET_SIZE[1]], .1, 'Weights')
                b2 = self.bias_variable(self.NET_SIZE[1], 'Biases')
                W_action = self.weight_variable([self.action_space, self.NET_SIZE[1]], .1, 'Action_Weights')
                h2 = tf.nn.relu(tf.add(tf.matmul(h1, W2), b2) + tf.matmul(input_action_pl, W_action))
            with tf.name_scope("Output"):
                W_out = self.weight_variable([self.NET_SIZE[1], 1], .1, 'Weights')
                b_out = self.bias_variable(1, 'Biases')
                out = tf.add(tf.matmul(h2, W_out), b_out)
                self.variables = [W1, b1, W2, b2, W_action, W_out, b_out]
                return out

    def get_variables(self):
        """
         Gets all weights and biases of the network
        Returns:
         A collection of all trainable variables
        """
        return self.variables

    def update_variables(self, v2):
        """
         Copies all trainable variables of this network into another network of
         same size
        Args:
         target_network: the network to be copied into
         sess: the tensorflow session that runs the assign-operations
        Returns:
         copy_ops: A list of operations to copy variables
        """
        v1 = self.get_variables()
        copy_ops = [v1[i].assign(tf.multiply(self.TAU, v2[i]) + tf.multiply((1-self.TAU), v1[i])) for i in range(len(v1))]
        return copy_ops

    def copy_to(self, target_network):
        """
         Copies all trainable variables of this network into another network of
         same size
        Args:
         target_network: the network to be copied into
         sess: the tensorflow session that runs the assign-operations
        """
        v1 = self.get_variables()
        v2 = target_network.get_variables()
        copy_ops = [v2[i].assign(v1[i]) for i in range(len(v1))]
        return copy_ops

import tensorflow as tf
from tensorflow.python.ops import rnn_cell

import numpy as np
import time

class Model():
  def __init__(self, args, infer=False):        
    self.args = args

    if infer:
      args.batch_size = 10
      args.seq_length = 347

    if args.model == 'rnn':
      cell_fn = rnn_cell.BasicRNNCell
    elif args.model == 'gru':
      cell_fn = rnn_cell.GRUCell
    elif args.model == 'lstm':
      cell_fn = rnn_cell.BasicLSTMCell
    else:
      raise Exception("model type not supported: {}".format(args.model))

  #self.t = time.time()
    self.learning_rate = 0.0001 # 0.001
    self.data_type = tf.float64
    self.n_steps = 347
    self.max_length = 347
    # Network Parameters
    self.n_input = 1 # MNIST data input (img shape: 28*28)
    self.n_hidden = 20 #128 # hidden layer num of features
    self.n_classes = 1 # MNIST total classes (0-9 digits)

    self.num_layers = 1
    self.state_size = self.n_hidden

  # tf Graph input
    self.graph_data_type = self.data_type #get_graph_data_type()

    self.x = tf.placeholder(self.graph_data_type, [args.batch_size, self.max_length, self.n_input], name="input_placeholder")
    self.y = tf.placeholder(self.graph_data_type, [args.batch_size, self.n_classes], name="label_placeholder")

    self.seq_length = tf.placeholder(tf.int32, [args.batch_size, 1], name="SequenceLength")

    #self.last_state = tf.placeholder(self.data_type, [None, 2*self.n_hidden], name="InputState") #state & cell => 2x n_hidden

    #self._x = tf.verify_tensor_all_finite(self.x, "X contains invalid data", name="XValidation")
    #self._y = tf.verify_tensor_all_finite(self.y, "Y contains invalid data", name="YValidation")

    # Define weights
    self.weights = {
        'hidden': tf.Variable(tf.random_normal([self.n_input, self.n_hidden], dtype=self.data_type)), #   Hidden layer weights
        'out': tf.Variable(tf.random_normal([self.n_hidden, self.n_classes], dtype=self.data_type))
    }
    self.biases = {
        'hidden': tf.Variable(tf.random_normal([self.n_hidden], dtype=self.data_type)),
        'out': tf.Variable(tf.random_normal([self.n_classes], dtype=self.data_type))
    }

    #pred, final_states = RNN(x, seq_length, istate, weights, biases)
    #_X = tf.verify_tensor_all_finite(_X, "-X contains invalid data???????", name="XValidation1")
    self._X = self.x
    self._weights = self.weights
    self._biases = self.biases

    # input shape: (batch_size, n_steps, n_input)
    #self._X = tf.transpose(self._X, [1, 0, 2])  # permute n_steps and batch_size
    #self._X = tf.unpack(self._X)
    # Reshape to prepare input to hidden activation
    #self._X = tf.reshape(self._X, [-1, self.n_input]) # (n_steps*batch_size, n_input)
    # Linear activation
    #self._X = tf.matmul(self._X, self._weights['hidden']) + self._biases['hidden']

    # Split to get a list of 'n_steps' tensors of shape (batch_size, n_input)
    #self._X = tf.split(0, self.n_steps, self._X)

    #_X = tf.verify_tensor_all_finite(_X, "-X contains invalid data!!!!!!!", name="XValidation2")

    # Define a lstm cell with tensorflow
    self.lstm_cell = lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(self.state_size, forget_bias=1.0) # , state_is_tuple=True

    # ('It took', 298.3749940395355, 'seconds to train for 3 epochs.')
    # http://r2rt.com/recurrent-neural-networks-in-tensorflow-ii.html
    self.lstm_cell = tf.nn.rnn_cell.MultiRNNCell([self.lstm_cell] * self.num_layers) #  , state_is_tuple=True

    # Split data because rnn cell needs a list of inputs for the RNN inner loop
    #self._X = tuple(tf.split(0, self.n_steps, self._X)) # n_steps * (batch_size, n_hidden)

    self.initial_state = lstm_cell.zero_state(args.batch_size, self.data_type)

    #print("state:: ", last_state)

    # Get lstm cell output
    # , sequence_length=_seq_length
    rnn_outputs, last_state = tf.nn.dynamic_rnn(self.lstm_cell, self._X, initial_state=self.initial_state)
    #_ = tf.nn.dynamic_rnn(self.lstm_cell, self._X, initial_state=self.initial_state)

    #reshape rnn_outputs and y so we can get the logits in a single matmul
    #rnn_outputs = tf.reshape(rnn_outputs, [-1, self.state_size])
    #y_reshaped = tf.reshape(self._y, [-1])

    # Linear activation
    # Get inner loop last output
    self.pred = tf.batch_matmul(y_reshaped, self._weights['out']) + self._biases['out']

    self.final_states = last_state
    self.out = rnn_outputs

    #pred = tf.verify_tensor_all_finite(pred_out, "Pred contains invalid data", name="PredValidation")

    # Define loss and optimizer
    # Mean squared error
    #self.reg_cost = tf.reduce_sum(1e-1 * (tf.nn.l2_loss(self.weights['out'])))
    #self.cost = tf.reduce_sum(tf.pow(self.pred-self.y, 2))/(2*self.max_length) + self.reg_cost

    #self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.cost) # Adam Optimizer

    # Evaluate model
    #self.correct_pred = tf.equal(tf.argmax(self.pred,1), tf.argmax(self.y,1))
    #self.accuracy = tf.reduce_mean(tf.cast(self.correct_pred, self.graph_data_type))

    #tf.scalar_summary('cost', self.cost)
    #tf.scalar_summary('reg_cost', self.reg_cost)

import numpy as np
import random
from random import shuffle
import tensorflow as tf
import pandas as pd
import time
import math

# Parameters
learning_rate = 0.0001 # 0.001
training_iters = 100000 #100000
batch_size = 128
display_step = 10


train_data_file = 'data/train.csv'
test_data_file = 'data/test.csv'

def prepare_data_sequence(data_sequence):
    vector_length = len(data_sequence)-1 # because we remove the prediction
    prediction = data_sequence.pop()
    return (data_sequence, prediction, vector_length)

def read_data_file(filename):
    data_frame = pd.read_csv(filename, delimiter=',', quotechar='"', header=1) # , nrows=3010
    data_matrix = data_frame.as_matrix()
    ids = data_matrix[:,0]
    values_str = data_matrix[:,1]
    vec_length = np.zeros((len(values_str)))
    values = []
    predictions = np.zeros((len(values_str)))
    max_length = 0
    i = 0
    for l in values_str:
	if l.find("inf") == -1:
        	parsed_values = l.replace('"', '').split(',')
        	sequence, prediction, vector_length = prepare_data_sequence(parsed_values)
		values.append(sequence)
		vec_length[i] = vector_length
		predictions[i] = prediction
        	if vector_length>max_length:
            		max_length = vector_length
        	i += 1
	else:
		print("found inf: ", l)

    predictions = predictions[..., np.newaxis]

    #print(predictions)

    c = 0
    for n in predictions:
	if n == np.inf:
		c += 1
		print("found inf: ", c)

    return (ids,values,predictions, vec_length, max_length)

def get_max_length(data_values_list):
    max_length = 0;
    for seq in data_values:
        cur_length = len(seq)
        if cur_length > max_length:
            max_length = cur_length
    return max_length

def prepare_training_matrix(data_values_list, max_sequence_length):
    training_sets = len(data_values_list)
    training_matrix = np.zeros((training_sets, max_sequence_length))
    i = 0
    for seq in data_values_list:
        seq_length = len(seq)
        training_matrix[i, 0:seq_length] = seq
        i += 1

    training_matrix = training_matrix[..., np.newaxis]
    return training_matrix

def tf_data_length(data):
    used = tf.sign(tf.reduce_max(tf.abs(data), reduction_indices=2))
    length = tf.reduce_sum(used, reduction_indices=1)
    length = tf.cast(length, tf.int32)
    return length

def get_input_dict(data_input, data_output, sequence_length, arg_inputs, arg_num_inputs):
	"""This function is suppose to return a dict which can be used to 
	initialize all the placeholders in the graph.
	"""
	in_data = {}
	count = len(data_output)
	print("count ",)
	for i in range(arg_num_inputs):
		v = arg_inputs[i]
		k = data_input[i].transpose()
		#print(i)
		in_data.update({inputs[i]: k})
	in_data.update({result: data_output[:arg_num_inputs]})
	#in_data.update({seq_length: sequence_length})
	in_data.update({num_inputs: arg_num_inputs})

	return in_data

def variable_summaries(var, name):
  """Attach a lot of summaries to a Tensor."""
  with tf.name_scope('summaries'):
    mean = tf.reduce_mean(var)
    tf.scalar_summary('mean/' + name, mean)
    with tf.name_scope('stddev'):
      stddev = tf.sqrt(tf.reduce_sum(tf.square(var - mean)))
    tf.scalar_summary('sttdev/' + name, stddev)
    tf.scalar_summary('max/' + name, tf.reduce_max(var))
    tf.scalar_summary('min/' + name, tf.reduce_min(var))
    tf.histogram_summary(name, var)


train_id, train_values, train_solutions, train_lengths, max_length = read_data_file(train_data_file)
training_matrix = prepare_training_matrix(train_values, max_length)

num_samples = len(train_lengths)
print("samples: ", num_samples)

NUM_EXAMPLES = 30000
num_test = 3000;

train_threshold = 10000 # good for float64
train_threshold = 1000 # good for float32

keep_count = 0.0
discard_count = 0.0

num_train = num_samples - num_test

num_batches = math.floor(num_train / batch_size)
num_train = int(batch_size * num_batches)

print('num_batches: ', num_batches)
print('num_train: ', num_train)

train_input  = np.zeros((num_train, max_length, 1))
train_output = np.zeros((num_train, 1))
train_length = np.zeros((num_train, 1))

for i in range(num_train):
    #if i % 1000 == 0:
        #print(i)
    max_value = train_solutions[i] #max(max(training_matrix[i]), train_solutions[i])
    do_keep = max_value < train_threshold
    if do_keep:
        train_input[i]  = (training_matrix[i])
        train_output[i] = (train_solutions[i])
        train_length[i] = (train_lengths[i])
        keep_count += 1.0
    else:
        discard_count += 1.0

print('discarded: ', discard_count/keep_count)

#train_input = training_matrix[:NUM_EXAMPLES]
#train_output = train_solutions[:NUM_EXAMPLES]
#train_length = train_lengths[:NUM_EXAMPLES]

test_input = training_matrix[NUM_EXAMPLES:NUM_EXAMPLES+NUM_EXAMPLES]
test_output = train_solutions[NUM_EXAMPLES:NUM_EXAMPLES+NUM_EXAMPLES]
test_length = train_lengths[:NUM_EXAMPLES]

val_input = training_matrix[2*NUM_EXAMPLES:3*NUM_EXAMPLES]
val_output = train_solutions[2*NUM_EXAMPLES:3*NUM_EXAMPLES]
val_length = train_lengths[:NUM_EXAMPLES]


print "test and training data loaded"
print('train input: ', len(train_input))
print('train output: ', len(train_output))
print('trains_length: ', len(train_output))

print_test_data = 50
#print('sequence: ', train_input[print_test_data])
#print('sequence next: ', train_output[print_test_data])


batch_size = 10
num_hidden = 24
frame_size = 1

train_count = batch_size

t = time.time()


# https://tensorhub.com/aymericdamien/tensorflow-rnn
# https://github.com/aymericdamien/TensorFlow-Examples/blob/master/notebooks/3_NeuralNetworks/recurrent_network.ipynb


'''
To classify images using a reccurent neural network, we consider every image row as a sequence of pixels.
Because MNIST image shape is 28*28px, we will then handle 28 sequences of 28 steps for every sample.
'''



# Network Parameters
n_input = 1 # MNIST data input (img shape: 28*28)
n_steps = max_length # timesteps
n_hidden = 128 # hidden layer num of features
n_classes = 1 # MNIST total classes (0-9 digits)

num_layers = 3

graph = tf.Graph()

with graph.as_default():
  # tf Graph input
  graph_data_type = tf.float64
  x = tf.placeholder(graph_data_type, [None, n_steps, n_input])
  seq_length = tf.placeholder(tf.int32, [batch_size, 1], name="SequenceLength")

  istate = tf.placeholder(graph_data_type, [batch_size, 2*n_hidden]) #state & cell => 2x n_hidden

  y = tf.placeholder(graph_data_type, [None, n_classes])

  _x = tf.verify_tensor_all_finite(x, "X contains invalid data", name="XValidation")
  _y = tf.verify_tensor_all_finite(y, "Y contains invalid data", name="YValidation")


  # Define weights
  weights = {
      'hidden': tf.Variable(tf.random_normal([n_input, n_hidden], dtype=graph_data_type)), #   Hidden layer weights
      'out': tf.Variable(tf.random_normal([n_hidden, n_classes], dtype=graph_data_type))
  }
  biases = {
      'hidden': tf.Variable(tf.random_normal([n_hidden], dtype=graph_data_type)),
      'out': tf.Variable(tf.random_normal([n_classes], dtype=graph_data_type))
  }


  def RNN(_X, _seq_length, _istate, _weights, _biases):

    #_X = tf.verify_tensor_all_finite(_X, "-X contains invalid data???????", name="XValidation1")

    # input shape: (batch_size, n_steps, n_input)
    _X = tf.transpose(_X, [1, 0, 2])  # permute n_steps and batch_size
    # Reshape to prepare input to hidden activation
    _X = tf.reshape(_X, [-1, n_input]) # (n_steps*batch_size, n_input)
    # Linear activation
    _X = tf.matmul(_X, _weights['hidden']) + _biases['hidden']

    #_X = tf.verify_tensor_all_finite(_X, "-X contains invalid data!!!!!!!", name="XValidation2")

    # Define a lstm cell with tensorflow
    # , state_is_tuple=True
    lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden, forget_bias=1.0, state_is_tuple=True)

    # ('It took', 298.3749940395355, 'seconds to train for 3 epochs.')
    # http://r2rt.com/recurrent-neural-networks-in-tensorflow-ii.html
    # , state_is_tuple=True
    lstm_cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell] * num_layers , state_is_tuple=True)

    # Split data because rnn cell needs a list of inputs for the RNN inner loop
    _X = tf.split(0, n_steps, _X) # n_steps * (batch_size, n_hidden)

    istate = lstm_cell.zero_state(batch_size, graph_data_type)
    #istate = lstm_cell.zero_state(128, tf.float64)

    # Get lstm cell output
    # , sequence_length=_seq_length
    outputs, states = tf.nn.rnn(lstm_cell, _X, initial_state=istate)

    # Linear activation
    # Get inner loop last output
    return tf.matmul(outputs[-1], _weights['out']) + _biases['out']


  pred = RNN(_x, seq_length, istate, weights, biases)

  #pred = tf.verify_tensor_all_finite(pred_out, "Pred contains invalid data", name="PredValidation")

  # Define loss and optimizer
  # Mean squared error
  tf_cost = tf.reduce_sum(tf.pow(pred-y, 2))/(2*max_length) + 1e-3 * (tf.nn.l2_loss(weights['out']))
  #cost = tf.Print(tf_cost, [pred, _y, tf_cost], 'Cost', summarize=1000)
  cost = tf.Print(tf_cost, [tf_cost], 'Cost', summarize=1000)

  #cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y)) # Softmax loss
  optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost) # Adam Optimizer

  # Evaluate model
  correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
  accuracy = tf.reduce_mean(tf.cast(correct_pred, graph_data_type))

print("It took", time.time() - t, "seconds to train for 3 epochs.")

# Launch the graph
with tf.Session(graph=graph) as sess:
    # Initializing the variables
    init = tf.initialize_all_variables()
    sess.run(init)

    ## Define summaries

    # Merge all the summaries and write them out to /tmp/mnist_logs (by default)
    merged = tf.merge_all_summaries()
    log_dir = 'log/train/' + str(num_layers)
    train_writer = tf.train.SummaryWriter(log_dir, sess.graph)

    step = 1
    ptr = 0
    # Keep training until reach max iterations
    while step * batch_size < training_iters:
        #batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        # Reshape data to get 28 seq of 28 elements
        #batch_xs = batch_xs.reshape((batch_size, n_steps, n_input))

	# inputs batch
	batch_xs = train_input[ptr:ptr+batch_size]

	# output batch
	batch_ys = train_output[ptr:ptr+batch_size]
	_seq_length = train_length[ptr:ptr+batch_size]

	ptr += batch_size

	#batch_xs = tf.Print(b_xs, [b_xs], 'BatchX')
	#batch_ys = tf.Print(b_ys, [b_ys], 'BatchY')
	#print('bx ', batch_xs.shape)
	#print('by ', batch_ys.shape)
	print("step ", step)

        # Fit training using batch data
        sess.run(optimizer, feed_dict={x: batch_xs, y: batch_ys, seq_length: _seq_length,
                                       istate: np.zeros((batch_size, 2*n_hidden))})
        if step % display_step == 0:
            # Calculate batch accuracy
            acc = sess.run(accuracy, feed_dict={x: batch_xs, y: batch_ys,
                                                istate: np.zeros((batch_size, 2*n_hidden))})
            # Calculate batch loss
            loss = sess.run(cost, feed_dict={x: batch_xs, y: batch_ys,
                                             istate: np.zeros((batch_size, 2*n_hidden))})

	    tf.scalar_summary('cost', loss)
            summary_str = sess.run(merged)
            train_writer.add_summary(summary_str, step)

            print "Iter " + str(step*batch_size) + ", Minibatch Loss= " + "{:.6f}".format(loss) + \
                  ", Training Accuracy= " + "{:.5f}".format(acc)
        step += 1
    print "Optimization Finished!"
    # Calculate accuracy for 256 mnist test images
    test_len = 256
    #test_data = mnist.test.images[:test_len].reshape((-1, n_steps, n_input))
    #test_label = mnist.test.labels[:test_len]
    test_data = test_input[:test_len]
    test_label = test_output[:test_len]

    test_res = sess.run(accuracy, 
			feed_dict={	x: test_data, 
					y: test_label,
                                        istate: np.zeros((test_len,2*n_hidden))
					}
			)

    print "Testing Accuracy:", test_res



    val_data = val_input[1]
    val_label = val_output[1]
    val_pred = sess.run(pred, feed_dict={	x: test_data, 
                                        istate: np.zeros((test_len,2*n_hidden))
					}
			)

    print('val data: ', val_data)
    print('val label: ', val_label)
    print('val pred: ', val_pred)

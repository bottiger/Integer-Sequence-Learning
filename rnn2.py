import numpy as np
import random
from random import shuffle
import tensorflow as tf
import pandas as pd
import time
import math
import pickle
import os

# Parameters
learning_rate = 0.0001 # 0.001
training_iters = 100000 #100000
batch_size = 128
display_step = 10

num_layers = 1 #6

# Network Parameters
n_input = 1 # MNIST data input (img shape: 28*28)
n_hidden = 20 #128 # hidden layer num of features
n_classes = 1 # MNIST total classes (0-9 digits)


batch_size = 10
num_hidden = 24
frame_size = 1

my_model_name = 'tmp/my-model.meta'

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


def get_training_data(name, amount, offset, force=False):
    set_filename = name + '.pickle'
    key_in         = 'input'
    key_out        = 'output'
    key_length     = 'length'
    key_max_length = 'max_length'

    if not os.path.exists(set_filename) or force:
        train_id, train_values, train_solutions, train_lengths, max_length = read_data_file(train_data_file)
        training_matrix = prepare_training_matrix(train_values, max_length)
        num_samples = len(train_lengths)
        print("samples: ", num_samples)
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

        #for i in range(num_train):
        for j in range(amount):
            i = j + offset
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

        dataset = {key_in: train_input, key_out: train_output, key_length: train_length, key_max_length: max_length}

        try:
            with open(set_filename, 'wb') as f:
                print('Persist pickle file: ', name)
                pickle.dump(dataset, f, pickle.HIGHEST_PROTOCOL)
        except Exception as e:
            print('Unable to save data to', set_filename, ':', e)

	print("return " + str(name) + " shape: " + str(train_output.shape))

        return train_input, train_output, train_length, max_length
    else:
        try:
          with open(set_filename, 'rb') as f:
	    print('Load persisted file: ', name)
            train_data_set = pickle.load(f)
            
	    train_input = train_data_set[key_in]
            train_output = train_data_set[key_out]
            train_length = train_data_set[key_length]
            max_length   = train_data_set[key_max_length]

	    return train_input, train_output, train_length, max_length
        except Exception as e:
          print('Unable to process data from', set_filename, ':', e)
          raise

def get_graph_data_type():
  return tf.float64;

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

    istate = lstm_cell.zero_state(batch_size, get_graph_data_type())
    #istate = lstm_cell.zero_state(128, tf.float64)

    # Get lstm cell output
    # , sequence_length=_seq_length
    outputs, states = tf.nn.rnn(lstm_cell, _X, initial_state=istate)

    # Linear activation
    # Get inner loop last output
    return tf.matmul(outputs[-1], _weights['out']) + _biases['out']

def persist_graph(graph_in, filename_in=my_model_name):
  # Launch the graph
  print('Preparing to export model')
  with tf.Session(graph=graph_in) as sess:

    # Initializing the variables
    init = tf.initialize_all_variables()
    sess.run(init)

    ptr = 0
    # inputs batch
    batch_xs = train_input[ptr:ptr+batch_size]

    # output batch
    batch_ys = train_output[ptr:ptr+batch_size]

    sess.run(optimizer, feed_dict={x: batch_xs, y: batch_ys, seq_length: _seq_length,
                                       istate: np.zeros((batch_size, 2*n_hidden))})

    print('exporting graph...')
    meta_graph_def = tf.train.export_meta_graph(filename=filename_in)
    print('exporting complete!')

def get_model_filename():
  return my_model_name + '.meta'

def have_persisted_model():
  return os.path.exists(get_model_filename())

NUM_EXAMPLES = 30000
num_test = 3000;
num_train = 20000;

# dataset has between 110720 and 113844 samples
train_input, train_output, train_length, max_length = get_training_data('train', num_train, 0)
test_input, test_output, test_length, max_length = get_training_data('test', 25000, 50000)
val_input, val_output, val_length, max_length = get_training_data('val', 25000, 75000)


#train_input = training_matrix[:NUM_EXAMPLES]
#train_output = train_solutions[:NUM_EXAMPLES]
#train_length = train_lengths[:NUM_EXAMPLES]

#test_input = training_matrix[NUM_EXAMPLES:NUM_EXAMPLES+NUM_EXAMPLES]
#test_output = train_solutions[NUM_EXAMPLES:NUM_EXAMPLES+NUM_EXAMPLES]
#test_length = train_lengths[:NUM_EXAMPLES]

#val_input = training_matrix[2*NUM_EXAMPLES:3*NUM_EXAMPLES]
#val_output = train_solutions[2*NUM_EXAMPLES:3*NUM_EXAMPLES]
#val_length = train_lengths[:NUM_EXAMPLES]


print "test and training data loaded"
print('train input: ', len(train_input))
print('train output: ', len(train_output))
print('trains_length: ', len(train_output))

print('train type: ', type(train_input))
print('val type: ', type(val_input))
print('train size: ', train_input.shape)
print('val size: ', val_input.shape)


print_test_data = 50
#print('sequence: ', train_input[print_test_data])
#print('sequence next: ', train_output[print_test_data])

n_steps = max_length # timesteps

train_count = batch_size


# https://tensorhub.com/aymericdamien/tensorflow-rnn
# https://github.com/aymericdamien/TensorFlow-Examples/blob/master/notebooks/3_NeuralNetworks/recurrent_network.ipynb


'''
To classify images using a reccurent neural network, we consider every image row as a sequence of pixels.
Because MNIST image shape is 28*28px, we will then handle 28 sequences of 28 steps for every sample.
'''


do_load_persisted_model = False #have_persisted_model()

graph = tf.Graph()

print('Generating the graph (~100 seconds pr layer)')
with graph.as_default():
  t = time.time()

  # tf Graph input
  graph_data_type = get_graph_data_type()
  x = tf.placeholder(graph_data_type, [None, n_steps, n_input], name="InputX")
  seq_length = tf.placeholder(tf.int32, [None, 1], name="SequenceLength")

  istate = tf.placeholder(graph_data_type, [None, 2*n_hidden], name="InputState") #state & cell => 2x n_hidden

  y = tf.placeholder(graph_data_type, [None, n_classes], name="InputY")

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

  pred = RNN(x, seq_length, istate, weights, biases)

  #pred = tf.verify_tensor_all_finite(pred_out, "Pred contains invalid data", name="PredValidation")

  # Define loss and optimizer
  # Mean squared error
  reg_cost = tf.reduce_sum(1e-1 * (tf.nn.l2_loss(weights['out'])))
  cost = tf.reduce_sum(tf.pow(pred-y, 2))/(2*max_length) + reg_cost
  #cost = tf.Print(tf_cost, [tf_cost], 'Cost', summarize=1000)

  #cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y)) # Softmax loss
  optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost) # Adam Optimizer

  # Evaluate model
  correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
  accuracy = tf.reduce_mean(tf.cast(correct_pred, graph_data_type))

  tf.scalar_summary('cost', cost)
  tf.scalar_summary('reg_cost', reg_cost)

  print("It took", time.time() - t, "seconds to train for " + str(num_layers) + " layers.")


#if not have_persisted_model():
  #graph = gen_graph(graph)
  #persist_graph(graph)
#else:
#  print('Skipping creating a graph, we should have one on disk')

# Add ops to save and restore all the variables.
#saver = tf.train.Saver()

# Launch the graph
with tf.Session(graph=graph) as sess:

    ## Define summaries
    # Merge all the summaries and write them out to /tmp/mnist_logs (by default)
    #tf.scalar_summary('cost', cost)
    merged = tf.merge_all_summaries()
    log_dir = 'log/train/' + str(num_layers)
    train_writer = tf.train.SummaryWriter(log_dir, sess.graph)

    # Initializing the variables
    init = tf.initialize_all_variables()
    sess.run(init)

    if do_load_persisted_model:
        t_start = time.time()
        print('Load persisted model')
        new_saver = tf.train.import_meta_graph(get_model_filename())
        new_saver.restore(sess, my_model_name)
        print("Persisted model loaded! ", time.time() - t, "seconds")
    #print('export graph')
    #meta_graph_def = tf.train.export_meta_graph(filename='tmp/my-model.meta')

    # Save the variables to disk.
    #save_path = saver.save(sess, "model/model.ckpt")
    #print("Model saved in file: %s" % save_path)

    #val_data = val_input[1]
    #val_data = val_data[np.newaxis, ...]
    #val_label = val_output[1]
    #val_label = val_label[np.newaxis, ...]
    #val_length = val_length[1]
    #val_length = val_length[np.newaxis, ...]
    #val_pred = sess.run(pred, feed_dict={	x: val_data, 
#						y: val_label, 
#						seq_length: val_length,
#                                        	istate: np.zeros((1,2*n_hidden))
					#}
#			)

 #   print('val data: ', val_data)
 #   print('val label: ', val_label)
 #   print('val pred: ', val_pred)

    step = 1
    ptr = 0
    # Keep training until reach max iterations
    while step < num_train/batch_size: #step * batch_size < training_iters:
        #batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        # Reshape data to get 28 seq of 28 elements
        #batch_xs = batch_xs.reshape((batch_size, n_steps, n_input))

	# inputs batch
	batch_xs = train_input[ptr:ptr+batch_size]

	# output batch
	batch_ys = train_output[ptr:ptr+batch_size]
	_seq_length = train_length[ptr:ptr+batch_size]

	ptr += batch_size

	#print("step ", step)

        # Fit training using batch data
        sess.run(optimizer, feed_dict={x: batch_xs, y: batch_ys, seq_length: _seq_length,
                                       istate: np.zeros((batch_size, 2*n_hidden))})
        if step % display_step == 0:
            # Calculate batch accuracy
            acc = sess.run(accuracy, feed_dict={x: batch_xs, y: batch_ys,
                                                istate: np.zeros((batch_size, 2*n_hidden))})

            val_xs = val_input[ptr:ptr+batch_size]
            val_ys = val_output[ptr:ptr+batch_size]
            val_seq_length = val_length[ptr:ptr+batch_size]

            # Calculate batch loss
            batch_loss = sess.run(cost, feed_dict={x: batch_xs, y: batch_ys,seq_length: _seq_length,
                                             istate: np.zeros((batch_size, 2*n_hidden))})

            batch_reg_loss = sess.run(reg_cost, feed_dict={x: batch_xs, y: batch_ys,seq_length: _seq_length,
                                             istate: np.zeros((batch_size, 2*n_hidden))})

            loss = sess.run(cost, feed_dict={x: val_xs, y: val_ys,seq_length: val_seq_length,
                                             istate: np.zeros((batch_size, 2*n_hidden))})

            reg_loss = sess.run(reg_cost, feed_dict={x: val_xs, y: val_ys,seq_length: val_seq_length,
                                             istate: np.zeros((batch_size, 2*n_hidden))})

	    
	    if loss<1000000:
                summary_str = sess.run(merged, feed_dict={x: val_xs, y: val_ys,
                                             istate: np.zeros((batch_size, 2*n_hidden))})
                train_writer.add_summary(summary_str, step)

            print "Iter " + str(step) + " of " + str(num_train/batch_size)
            print "Batch cost= " + "{:.6f}".format(batch_loss) + ", Reg cost= " + "{:.6f}".format(batch_reg_loss)
            print "Validation cost= " + "{:.6f}".format(loss) + ", Reg cost= " + "{:.6f}".format(reg_loss)
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



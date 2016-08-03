#Source code with the blog post at http://monik.in/a-noobs-guide-to-implementing-rnn-lstm-using-tensorflow/

import numpy as np
import random
from random import shuffle
import tensorflow as tf
import pandas as pd

train_data_file = 'data/train.csv'
test_data_file = 'data/test.csv'

def prepare_data_sequence(data_sequence):
    vector_length = len(data_sequence)-1 # because we remove the prediction
    prediction = data_sequence.pop()
    return (data_sequence, prediction, vector_length)

def read_data_file(filename):
    data_frame = pd.read_csv(filename, delimiter=',', quotechar='"', header=1, nrows=3010)
    data_matrix = data_frame.as_matrix()
    ids = data_matrix[:,0]
    values_str = data_matrix[:,1]
    vec_length = np.zeros((len(values_str)))
    values = []
    predictions = np.zeros((len(values_str)))
    max_length = 0
    i = 0
    for l in values_str:
        parsed_values = l.replace('"', '').split(',')
        sequence, prediction, vector_length = prepare_data_sequence(parsed_values)
	values.append(sequence)
	vec_length[i] = vector_length
	predictions[i] = prediction
        if vector_length>max_length:
            max_length = vector_length
        i += 1

    predictions = predictions[..., np.newaxis]
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

train_id, train_values, train_solutions, train_lengths, max_length = read_data_file(train_data_file)
training_matrix = prepare_training_matrix(train_values, max_length)

# Not for testing while developing
#test_id, test_values, test_solutions, max_length_test = read_data_file(test_data_file)
#testing_matrix = prepare_training_matrix(test_values, max_length)



NUM_EXAMPLES = 100

train_input = training_matrix[:NUM_EXAMPLES]
train_output = train_solutions[:NUM_EXAMPLES]
train_length = train_lengths[:NUM_EXAMPLES]

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


batch_size = 2
num_hidden = 24
frame_size = 1

train_count = batch_size

cell = tf.nn.rnn_cell.BasicLSTMCell(num_hidden, state_is_tuple=True)

num_inputs = tf.placeholder(tf.int32, name='NumInputs')

inputs = [tf.placeholder(tf.float32,shape=[1, max_length], name='InputData') for _ in range(batch_size)]
result = tf.placeholder(tf.float32, shape=[batch_size, 1], name='OutputData')


#inputs = tf.Print(tf_inputs, [tf_inputs, len(tf_inputs)], 'Inputs: ')
#seq_length = tf.Print(tf_seq_length, [tf_seq_length, tf_seq_length.get_shape()], 'SequenceLength: ')

# , sequence_length=tf_data_length(data)
# , sequence_length=seq_length
outputs, states = tf.nn.rnn(cell, inputs, dtype=tf.float32) 

#o = tf.Print(outputs, [outputs, states], 'RNN Output')

#outputs = tf.Print(tf_outputs, [tf_outputs], 'RNN outputs: ')

# tf.to_float(o.get_shape()[0]) #
#outputs_last = outputs[-1]   #we actually only need the last output from the model, ie: last element of outputs

outputs2 = tf.Print(outputs, [outputs], 'Last: ', name="Last", summarize=800)

tf_weight = tf.Variable(tf.truncated_normal([batch_size, num_hidden, frame_size]), name='Weight')
tf_bias   = tf.Variable(tf.constant(0.1, shape=[batch_size]), name='Bias')

weight = tf.Print(tf_weight, [tf_weight, tf_weight.get_shape()], "Weight: ")
bias = tf.Print(tf_bias, [tf_bias, tf_bias.get_shape()], "bias: ")

#print('last ', last.get_shape())
print('bias: ', bias.get_shape())
print('weight: ', weight.get_shape())
print('targets ', result.get_shape())
print('RNN input ', type(inputs))
print('RNN input len()', len(inputs))
print('RNN input[0] ', inputs[0].get_shape())
#print('RNN output ', outputs2.get_shape())

tf_prediction = tf.batch_matmul(outputs2, weight) + bias
prediction = tf.Print(tf_prediction, [tf_prediction, tf_prediction.get_shape()], 'prediction: ')

#tf_result = tf.Print(result, [result], 'result: ')
tf_result = result

print('prediction ', prediction.get_shape())

print('train input: ', train_input.shape)
print('train output: ', train_output.shape)

print('test input: ', test_input.shape)
print('test output: ', test_output.shape)

tf_pow_loss = tf.pow(tf_result-prediction,2)
pow_loss = tf.Print(tf_pow_loss, [tf_pow_loss], 'tf_pow_loss ', summarize=batch_size*8)

tf_l2_loss = tf.reduce_mean(pow_loss)    #compute the cost for this batch of data
#l2_loss = tf.Print(tf_l2_loss, [tf_l2_loss], 'Loss ')

#sm = tf.nn.softmax_cross_entropy_with_logits(tf_result, prediction)
#cost = tf.reduce_mean(sm)

cost = tf.nn.l2_loss(tf_result - prediction)

#optimizer = tf.train.AdamOptimizer()
learning_rate  = 0.05
optimizer = tf.train.GradientDescentOptimizer(learning_rate)


minimize = optimizer.minimize(cost)

mistakes = tf.not_equal(tf.argmax(result, 1), tf.argmax(prediction, 1))
error = tf.reduce_mean(tf.cast(mistakes, tf.float32))

init_op = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init_op)

no_of_batches = 3#int(len(train_input)) / batch_size
epoch = 1

val_dict = get_input_dict(val_input, val_output, train_length, inputs, batch_size)

for i in range(epoch):
    ptr = 0
    for j in range(no_of_batches):

	print('eval w: ', weight.eval(session=sess))

	#in_data = {}
	#for i in range(train_count):
	#	v = inputs[i]
	#	k = train_input[i].transpose()
	#	print(i)
	#	in_data.update({inputs[i]: k})
	#in_data.update({result: train_output})

	t_i = train_input[ptr:ptr+batch_size]
	t_o = train_output[ptr:ptr+batch_size]
	t_l = train_length[ptr:ptr+batch_size]

	sess.run(minimize,feed_dict=get_input_dict(t_i, t_o, t_l, inputs, batch_size))

	ptr += batch_size

	#print('eval outputs: ', outputs.eval(session=sess))

	#print("result dim: ", train_output.shape)
	#print('eval w2: ', weight.eval(session=sess))
	print("result: ", tf_result)
	print("result len: ", tf_result.get_shape())
	print("prediction: ", prediction)
	print("prediction len: ", prediction.get_shape())


	c_val = sess.run(error, feed_dict = val_dict )
	print "Validation cost: {}, on Epoch {}".format(c_val,i)


    print "Epoch ",str(i)

print('test input: ', type(test_input))
print('test output: ', type(test_output))

incorrect = sess.run(error,get_input_dict(test_input, test_output, test_length, inputs, batch_size))

print('incorrect: ', incorrect)

print('eval w: ', weight.eval(session=sess))
print('eval b: ', bias.eval(session=sess))
#print('eval val: ', val.eval(session=sess))
#print('eval val: ', l2_loss.eval(session=sess))


# Manual testing

test_id = 3
test_seq = test_input[test_id]
test_seq = np.expand_dims(test_seq, axis=1)

test_res = test_output[test_id]

#print('test seq: ', test_seq)
print('test res', test_res)

#print('result', sess.run(prediction,get_input_dict(test_seq, test_res, len(test_seq), inputs, batch_size)))
#print('Epoch {:2d} error {:3.1f}%'.format(i + 1, 100 * incorrect))

#print sess.run(prediction,{data: [[[1],[0],[0],[1],[1],[0],[1],[1],[1],[0],[1],[0],[0],[1],[1],[0],[1],[1],[1],[0]]]})
#print('Epoch {:2d} error {:3.1f}%'.format(i + 1, 100 * incorrect))

sess.close()

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
    data_frame = pd.read_csv(filename, delimiter=',', quotechar='"', header=1)
    data_matrix = data_frame.as_matrix()
    ids = data_matrix[:,0]
    values_str = data_matrix[:,1]
    values = []
    predictions = np.zeros((len(values_str)))
    max_length = 0
    i = 0
    for l in values_str:
        parsed_values = l.replace('"', '').split(',')
        sequence, prediction, vector_length = prepare_data_sequence(parsed_values)
	values.append(sequence)
	predictions[i] = prediction
        if vector_length>max_length:
            max_length = vector_length
        i += 1

    predictions = predictions[..., np.newaxis]
    return (ids,values,predictions, max_length)

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


train_id, train_values, train_solutions, max_length = read_data_file(train_data_file)
training_matrix = prepare_training_matrix(train_values, max_length)

# Not for testing while developing
#test_id, test_values, test_solutions, max_length_test = read_data_file(test_data_file)
#testing_matrix = prepare_training_matrix(test_values, max_length)



NUM_EXAMPLES = 1000

#train_input = ['{0:020b}'.format(i) for i in range(NUM_EXAMPLES*2)]
#shuffle(train_input)
#train_input = [map(int,i) for i in train_input]
#ti  = []
#train_output = []

#for i in train_input:
#    temp_list = []
#    count = 0
#    for j in i:
#      temp_list.append([j])
#      if j == 1:
#        count += 1

#    train_output.append(count)      
#    ti.append(np.array(temp_list))

#train_input = ti


#test_input = train_input[NUM_EXAMPLES:]
#test_output = train_output[NUM_EXAMPLES:]
#train_input = train_input[:NUM_EXAMPLES]
#train_output = train_output[:NUM_EXAMPLES]

train_input = training_matrix[:NUM_EXAMPLES]
train_output = train_solutions[:NUM_EXAMPLES]

test_input = training_matrix[NUM_EXAMPLES:NUM_EXAMPLES+NUM_EXAMPLES]
test_output = train_solutions[NUM_EXAMPLES:NUM_EXAMPLES+NUM_EXAMPLES]

#test_input = testing_matrix
#test_output = test_solutions


print "test and training data loaded"
print('train input: ', len(train_input))
print('train output: ', len(train_output))


batch_size = 100

frame_size = 1

data = tf.placeholder(tf.float32, shape=(None, max_length, 1), name='InputData') #Number of examples, number of input, dimension of each input
target = tf.placeholder(tf.float32, shape=(None, 1), name='TargetData')


#print('target shape: ', target.get_shape())
print('data shape', data.get_shape())
#print('int(shape) ', int(target.get_shape()[1]))

num_hidden = 24

cell = tf.nn.rnn_cell.BasicLSTMCell(num_hidden, state_is_tuple=True)
#val, _ = tf.nn.dynamic_rnn(cell, data, dtype=tf.float32, sequence_length=tf_data_length(data))
val, _ = tf.nn.dynamic_rnn(cell, data, dtype=tf.float32)

val = tf.transpose(val, [1, 0, 2])
#val = tf.transpose(val, [0, 1, 5, 5 ,5])

print('val shape, ', val.get_shape())

last = tf.gather(val, int(val.get_shape()[0]) - 1)

weight = tf.Variable(tf.truncated_normal([num_hidden, batch_size]), name='Weight')
bias = tf.Variable(tf.constant(0.1, shape=[1]), name='Bias')

#prediction = tf.nn.softmax(tf.matmul(last, weight) + bias)
prediction = tf.matmul(last, weight) + bias

print('last ', last.get_shape())
print('bias: ', bias.get_shape())
print('weight: ', weight.get_shape())
print('targets ', target.get_shape())
print('prediction ', prediction.get_shape())

print('train input: ', train_input.shape)
print('train output: ', train_output.shape)

print('test input: ', test_input.shape)
print('test output: ', test_output.shape)

#dist = tf.concat(1, [target, prediction])

#l2_loss = -tf.reduce_sum(target - prediction) # tf.nn.l2_loss
l2_loss = tf.nn.l2_loss(target - prediction)
optimizer = tf.train.AdamOptimizer()
minimize = optimizer.minimize(l2_loss)

mistakes = tf.not_equal(tf.argmax(target, 1), tf.argmax(prediction, 1))
error = tf.reduce_mean(tf.cast(mistakes, tf.float32))

init_op = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init_op)

no_of_batches = int(len(train_input)) / batch_size
epoch = 1 #500

for i in range(epoch):
    ptr = 0
    for j in range(no_of_batches):
        inp, out = train_input[ptr:ptr+batch_size], train_output[ptr:ptr+batch_size]

	#inp = inp[..., np.newaxis]

        #print('inp: ', inp)
        #print('inp type: ', type(inp))
        #print('inp shape: ', inp.shape)
        #print('out: ', out)
        ptr+=batch_size
        sess.run(minimize,{data: inp, target: out})
    print "Epoch ",str(i)

print('test input: ', type(test_input))
print('test output: ', type(test_output))

incorrect = sess.run(error,{data: test_input, target: test_output})

print('incorrect: ', incorrect)

#print sess.run(prediction,{data: [[[1],[0],[0],[1],[1],[0],[1],[1],[1],[0],[1],[0],[0],[1],[1],[0],[1],[1],[1],[0]]]})
#print('Epoch {:2d} error {:3.1f}%'.format(i + 1, 100 * incorrect))

sess.close()

#Source code with the blog post at http://monik.in/a-noobs-guide-to-implementing-rnn-lstm-using-tensorflow/

import numpy as np
import random
from random import shuffle
import tensorflow as tf

NUM_EXAMPLES = 1000

train_input = ['{0:020b}'.format(i) for i in range(NUM_EXAMPLES*2)]
shuffle(train_input)
train_input = [map(int,i) for i in train_input]
ti  = []
train_output = []

for i in train_input:
    temp_list = []
    count = 0
    for j in i:
      temp_list.append([j])
      if j == 1:
        count += 1

    train_output.append(count)      
    ti.append(np.array(temp_list))

train_input = ti

#for i in train_input:
#    count = 0
#    for j in i:
#        if j[0] == 1:
#            count+=1
    #temp_list = ([0]*21)
    #temp_list[count]=1
    #train_output.append(temp_list)
#    train_output.append(count)

test_input = train_input[NUM_EXAMPLES:]
test_output = train_output[NUM_EXAMPLES:]
train_input = train_input[:NUM_EXAMPLES]
train_output = train_output[:NUM_EXAMPLES]


print "test and training data loaded"
print('train input: ', len(train_input))


batch_size = 100

#target = tf.placeholder(tf.float32, [None, 1])

#print('t shape: ,', t_array.get_shape())

target = tf.placeholder(tf.float32, shape=(None, ), name='TargetData')
data = tf.placeholder(tf.float32, [None, 20,1], name='InputData') #Number of examples, number of input, dimension of each input
#target = tf.placeholder(tf.float32, [None, 1])

#print('target shape: ', target.get_shape())
print('data shape', data.get_shape())
#print('int(shape) ', int(target.get_shape()[1]))

num_hidden = 24
cell = tf.nn.rnn_cell.LSTMCell(num_hidden)
val, _ = tf.nn.dynamic_rnn(cell, data, dtype=tf.float32)
val = tf.transpose(val, [1, 0, 2])

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

#dist = tf.concat(1, [target, prediction])
l2_loss = -tf.reduce_sum(target - prediction) # tf.nn.l2_loss
optimizer = tf.train.AdamOptimizer()
minimize = optimizer.minimize(l2_loss)

mistakes = tf.not_equal(tf.argmax(target, 1), tf.argmax(prediction, 1))
error = tf.reduce_mean(tf.cast(mistakes, tf.float32))

init_op = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init_op)

no_of_batches = int(len(train_input)) / batch_size
epoch = 500

for i in range(epoch):
    ptr = 0
    for j in range(no_of_batches):
        inp, out = train_input[ptr:ptr+batch_size], train_output[ptr:ptr+batch_size]
        ptr+=batch_size
        sess.run(minimize,{data: inp, target: out})
    print "Epoch ",str(i)

incorrect = sess.run(error,{data: test_input, target: test_output})

#print sess.run(prediction,{data: [[[1],[0],[0],[1],[1],[0],[1],[1],[1],[0],[1],[0],[0],[1],[1],[0],[1],[1],[1],[0]]]})
#print('Epoch {:2d} error {:3.1f}%'.format(i + 1, 100 * incorrect))

sess.close()

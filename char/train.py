from __future__ import print_function
import numpy as np
import tensorflow as tf

import argparse
import time
import os
from six.moves import cPickle

import pandas as pd
import math
import pickle

from utils import TextLoader
from model import Model

train_data_file = '../data/train.csv'
test_data_file = '../data/test.csv'

num_test = 3000;

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

def get_training_data(args, name, amount, offset, force=False):
    set_filename = name + '.pickle'
    key_in         = 'input'
    key_out        = 'output'
    key_length     = 'length'
    key_max_length = 'max_length'

    if not os.path.exists(set_filename) or force:
        batch_size = args.batch_size

        train_id, train_values, train_solutions, train_lengths, max_length = read_data_file(train_data_file)
        training_matrix = prepare_training_matrix(train_values, max_length)
        num_samples = len(train_lengths)
        print("samples: ", num_samples)
        train_threshold = 10000000000000000000000000000 # good for float64
        #train_threshold = 1000 # good for float32
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
        nextline = 0;
        for j in range(amount):
            i = j + offset
            #if i % 1000 == 0:
                #print(i)
            max_value = train_solutions[i] #max(max(training_matrix[i]), train_solutions[i])
            do_keep = max_value < train_threshold
            if do_keep:
                train_input[nextline]  = (training_matrix[i])
                train_output[nextline] = (train_solutions[i])
                train_length[nextline] = (train_lengths[i])
		nextline += 1
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

def get_model_filename():
  return my_model_name + '.meta'

def have_persisted_model():
  return os.path.exists(get_model_filename())

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data/tinyshakespeare',
                       help='data directory containing input.txt')
    parser.add_argument('--save_dir', type=str, default='save',
                       help='directory to store checkpointed models')
    parser.add_argument('--rnn_size', type=int, default=128,
                       help='size of RNN hidden state')
    parser.add_argument('--num_layers', type=int, default=2,
                       help='number of layers in the RNN')
    parser.add_argument('--model', type=str, default='lstm',
                       help='rnn, gru, or lstm')
    parser.add_argument('--batch_size', type=int, default=50,
                       help='minibatch size')
    parser.add_argument('--seq_length', type=int, default=50,
                       help='RNN sequence length')
    parser.add_argument('--num_epochs', type=int, default=50,
                       help='number of epochs')
    parser.add_argument('--save_every', type=int, default=1000,
                       help='save frequency')
    parser.add_argument('--grad_clip', type=float, default=5.,
                       help='clip gradients at this value')
    parser.add_argument('--learning_rate', type=float, default=0.2, # 0.002
                       help='learning rate')
    parser.add_argument('--decay_rate', type=float, default=0.97,
                       help='decay rate for rmsprop')
    parser.add_argument('--init_from', type=str, default=None,
                       help="""continue training from saved model at this path. Path must contain files saved by previous training process:
                            'config.pkl'        : configuration;
                            'chars_vocab.pkl'   : vocabulary definitions;
                            'checkpoint'        : paths to model file(s) (created by tf).
                                                  Note: this file contains absolute paths, be careful when moving files around;
                            'model.ckpt-*'      : file(s) with model definition (created by tf)
                        """)
    args = parser.parse_args()
    train(args)

def train(args):

    display_step = 100
    num_train = 20000;
    train_input, train_output, train_length, max_length = get_training_data(args, 'train', num_train, 0)
    test_input, test_output, test_length, max_length = get_training_data(args, 'test', 25000, 50000)
    val_input, val_output, val_length, max_length = get_training_data(args, 'val', 25000, 75000)

    #for i in range(2):
    #  print('i: ' + str(i) + ' => ' + str(train_input[i,:]))

    train_input = train_input.astype(int)

    data_loader = TextLoader(args.data_dir, args.batch_size, args.seq_length)
    args.vocab_size = 50000 #data_loader.vocab_size

    # check compatibility if training is continued from previously saved model
    if args.init_from is not None:
        # check if all necessary files exist
        assert os.path.isdir(args.init_from)," %s must be a a path" % args.init_from
        assert os.path.isfile(os.path.join(args.init_from,"config.pkl")),"config.pkl file does not exist in path %s"%args.init_from
        assert os.path.isfile(os.path.join(args.init_from,"chars_vocab.pkl")),"chars_vocab.pkl.pkl file does not exist in path %s" % args.init_from
        ckpt = tf.train.get_checkpoint_state(args.init_from)
        assert ckpt,"No checkpoint found"
        assert ckpt.model_checkpoint_path,"No model path found in checkpoint"

        # open old config and check if models are compatible
        with open(os.path.join(args.init_from, 'config.pkl')) as f:
            saved_model_args = cPickle.load(f)
        need_be_same=["model","rnn_size","num_layers","seq_length"]
        for checkme in need_be_same:
            assert vars(saved_model_args)[checkme]==vars(args)[checkme],"Command line argument and saved model disagree on '%s' "%checkme

        # open saved vocab/dict and check if vocabs/dicts are compatible
        with open(os.path.join(args.init_from, 'chars_vocab.pkl')) as f:
            saved_chars, saved_vocab = cPickle.load(f)
        assert saved_chars==data_loader.chars, "Data and loaded model disagreee on character set!"
        assert saved_vocab==data_loader.vocab, "Data and loaded model disagreee on dictionary mappings!"

    with open(os.path.join(args.save_dir, 'config.pkl'), 'wb') as f:
        cPickle.dump(args, f)
    with open(os.path.join(args.save_dir, 'chars_vocab.pkl'), 'wb') as f:
        cPickle.dump((data_loader.chars, data_loader.vocab), f)

    model = Model(args)

    print("num_layers: ", args.num_layers)

    with tf.Session() as sess:
        tf.initialize_all_variables().run()
        saver = tf.train.Saver(tf.all_variables())
        # restore model
        if args.init_from is not None:
            saver.restore(sess, ckpt.model_checkpoint_path)
        for e in range(args.num_epochs):
            sess.run(tf.assign(model.lr, args.learning_rate * (args.decay_rate ** e)))
            data_loader.reset_batch_pointer()
            state = model.initial_state.eval()

            step = 0
            ptr = 0

	    print('train_input: ', train_input.shape)

            while step < num_train/args.batch_size:
                b = step
            #for b in range(data_loader.num_batches):
		step += 1
                start = time.time()

	        # inputs batch
	        x = np.squeeze(train_input[ptr:ptr+args.batch_size, :args.batch_size])

	        # output batch
	        y = np.squeeze(train_input[ptr:ptr+args.batch_size, 1:args.batch_size+1])
		ptr += args.batch_size+1
                #x, y = data_loader.next_batch()
		#print('x: ', x.shape)
		#print('y: ', y.shape)
		#print('x: ', x[1])
		#print('y: ', y)
                feed = {model.input_data: x, model.targets: y, model.initial_state: state}
                tt, calc_res, reg_cost, train_loss, state, _ = sess.run([model.target_vector, model.logits, model.reg_cost, model.cost, model.final_state, model.train_op], feed)
		print('out len: ', len(tt))
		print('target: ', tt)
		print('calc_res: ', calc_res)
                end = time.time()
                print("{}/{} (epoch {}), train_loss = {:.3f}, time/batch = {:.3f}, reg_cost = {:.3f}" \
                    .format(e * data_loader.num_batches + b,
                            args.num_epochs * data_loader.num_batches,
                            e, train_loss, end - start, reg_cost))
                if (e * data_loader.num_batches + b) % args.save_every == 0\
                    or (e==args.num_epochs-1 and b == data_loader.num_batches-1): # save for the last result
                    checkpoint_path = os.path.join(args.save_dir, 'model.ckpt')
                    saver.save(sess, checkpoint_path, global_step = e * data_loader.num_batches + b)
                    print("model saved to {}".format(checkpoint_path))

		if step % display_step == 0:
		    print('x: ', x[1])

if __name__ == '__main__':
    main()

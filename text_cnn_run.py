#feed_dict rewrites the value of tensors in the graph
# implement updating vocab
# make adadelta work
# look up adadelta source code
# look up minibatch handling
# specify batch size, squeeze dev/train totals to fit
# use Higher Order Operations/TensorArrays?

#other notes:
    # Adam does got lower softmax, by a couple %
    # dev softmax min is before dev accuracy max -- typically after epoch 2-4 (zero-based)
    # should we optimize for softmax min or accuracy max??

#minor changes:
#shuffled batches each epoch

#program expects:
    # flags: -a for Adagrad, -s for short (test run)
    # argv[1] directory with train.data, dev.data, train.labels, dev.labels in SST format
    # argv[2] learning rate
    # argv[3] number of epochs
    # argv[4] identifier tag (appended to filename to distinguish multiple runs)

#outputs in file named
    # directory, Optimizer name, number of epochs, identifier .txt
    # (commas only where necessary to distinguish numbers)
        # initial accuracy (train data)
        # training and dev accuracy at each epoch, dev softmax accuracy

import tensorflow as tf
import random
import linecache
from text_cnn_methods import *
import numpy as np
import sys, getopt
import os

#define hyperparameters
def define_globals(args):
    params = {'WORD_VECTOR_LENGTH' : 300,
        'FILTERS' : 100,
        'KERNEL_SIZES' : [3,4,5],
        'CLASSES' : 2,
        'MAX_LENGTH' : 59,

        'L2_NORM_CONSTRAINT' : 3.0,
        'TRAIN_DROPOUT' : 0.5,

        'BATCH_SIZE' : 50,
        'EPOCHS' : args[2],

        'Adagrad' : False,
        'LEARNING_RATE' : args[1],
        'TRAIN_FILE_NAME' : 'train',
        'DEV_FILE_NAME' : 'dev',
        'WORD_VECS_FILE_NAME' : 'output.txt',
        'OUTPUT_FILE_NAME' : args[0],
        'SST' : True,
        'DEV' : True,

        #set by program-do not change!
        'batch_iterations' : 0,
        'epoch' : 1,
        'l2-loss' : tf.constant(0),
        #debug
        'key_errors' : [],
        'changes' : 0}
    return params


#get random batch of examples from test file
def get_batches(lines, params, train_x, train_y):
    if params['epoch'] == 1:
        np.random.seed(3435)
        # print train_x.shape, train_y.shape
        if train_y.shape[0] % params['BATCH_SIZE'] > 0:
            extra_data_num = params['BATCH_SIZE'] - train_y.shape[0] % params['BATCH_SIZE']
            train_set_x, train_set_y = shuffle_in_unison(train_x, train_y)
            extra_data_x = train_set_x[:extra_data_num]
            extra_data_y = train_set_y[:extra_data_num]
            new_data_x = np.append(train_x, extra_data_x, axis=0)
            new_data_y = np.append(train_y, extra_data_y, axis=0)
            # print new_data_x.shape, new_data_y.shape
        else:
            new_data_x = train_x
            new_data_y = train_y
    new_data_x, new_data_y = shuffle_in_unison(new_data_x, new_data_y)
    return new_data_x, new_data_y

#index and loop through same batches again
def get_batch(batches_x, batches_y, index, d, params):
    cur_batch_x = batches_x[index*params['BATCH_SIZE']:(index+1)*params['BATCH_SIZE'],:]
    cur_batch_y = batches_y[index*params['BATCH_SIZE']:(index+1)*params['BATCH_SIZE'],:]
    return sub_vectors(cur_batch_x, d, params), cur_batch_y

def sum_prob(x, y_,set_x, set_y, keys, params, correct_prediction, accuracy, dropout, sess):
        set_x_temp = set_x
        set_y_temp = set_y
        examples_correct = 0
        while set_y_temp.shape[0] >= params['BATCH_SIZE']:
            examples_correct += accuracy.eval(feed_dict={x: sub_vectors(set_x_temp[0:50], keys, params), y_: set_y_temp[0:50], dropout: 1.0}, session = sess)
            set_x_temp = set_x_temp[50:]
            set_y_temp = set_y_temp[50:]
        if set_y_temp.shape[0] > 0:
            remaining_examples = set_y_temp.shape[0]
            set_y_temp = np.append(set_y_temp, np.zeros((params['BATCH_SIZE'] - remaining_examples, params['CLASSES'])))
            """
            add1 = ['<PAD>'] * params['MAX_LENGTH']
            print add1
            print type(add1)
            for i in range(params['BATCH_SIZE'] - remaining_examples - 1):
                add1= add1.append(['<PAD>'] * params['MAX_LENGTH'])
            print add1
            print 'again' , set_x_temp
            set_x_temp = np.append(set_x_temp, np.asarray(add1))
            print set_x_temp
            """
            zeroes = np.zeros((params['BATCH_SIZE']-remaining_examples, params['MAX_LENGTH'], params['WORD_VECTOR_LENGTH']))
            print zeroes.shape
            print tf.reduce_mean(tf.slice(tf.cast(correct_prediction, tf.float32), tf.constant([0]),
                tf.constant([set_y_temp.shape[0]]))).eval(
                feed_dict={x: np.append(sub_vectors(set_x_temp[0:remaining_examples], keys, params), zeroes),
                y_: set_y_temp, dropout: 1.0}, session = sess).eval()
            print tf.reduce_mean(tf.cast(correct_prediction, tf.float32)).eval(feed_dict={x: np.append(sub_vectors(set_x_temp[0:remaining_examples], keys, params), zeroes), y_: set_y_temp, dropout: 1.0}, session = sess).eval()
            examples_correct += tf.reduce_mean(tf.slice(tf.cast(correct_prediction, tf.float32), tf.constant([0]), tf.constant([set_y_temp.shape[0]]))).eval(feed_dict={x: sub_vectors(set_x_temp[0:remaining_examples], keys, params), y_: set_y_temp, dropout: 1.0}, session = sess)
        return examples_correct*50/set_y.shape[0]

def main(argv):
    try:
        opts, args = getopt.getopt(argv, "slfa")
    except getopt.GetoptError:
        print('Unable to run; GetoptError')
        sys.exit(2)
    try:
        args[1] = float(args[1])
        args[2] = int(args[2])
    except SyntaxError:
        print args[1], "type", type(args[1])
        print('Unable to run; command line input does not match')
        sys.exit(2)
    params = define_globals(args)
    if args[0] == 'sst1':
        params['CLASSES'] = 5
    for opt in opts:
        if opt[0] == ("-a"):
            params['Adagrad'] = True
            params['OUTPUT_FILE_NAME'] += 'Adagrad'
            break
    if params['Adagrad'] == False:
        params['OUTPUT_FILE_NAME'] += 'Adam'
    params['OUTPUT_FILE_NAME'] += str(params['LEARNING_RATE'])
    for opt in opts:
        if opt[0] == ("-s"):
            params['TRAIN_FILE_NAME'] = 'test-short'
            params['DEV_FILE_NAME'] = 'dev-short'
            params['WORD_VECS_FILE_NAME'] = 'output-short.txt'
            params['OUTPUT_FILE_NAME'] += 's'
    params['OUTPUT_FILE_NAME'] += ',' + str(params['EPOCHS']) + ',' + args[3]
    output = open(params['OUTPUT_FILE_NAME'] + '.txt', 'a', 0)
    if params['Adagrad']:
        output.write("Running Adagrad with a learning rate of ")
    else:
        output.write("Running Adam with a learning rate of ")
    output.write(str(params['LEARNING_RATE']) + ' and ' + str(params['EPOCHS']) + ' epochs\n')
    output.write("Using files: " + str(params['TRAIN_FILE_NAME']) + ', '
        + str(params['DEV_FILE_NAME']) + ', '
        + str(params['WORD_VECS_FILE_NAME']) + '\n')

    train_size = find_lines(params['TRAIN_FILE_NAME'] + '.labels')
    dev_size = find_lines(params['DEV_FILE_NAME'] + '.labels')

    #create training and eval sets
    dev_x, dev_y = get_all(args[0], params['DEV_FILE_NAME'], dev_size, params)
    train_x, train_y = get_all(args[0], params['TRAIN_FILE_NAME'], train_size, params)

    vocab = find_vocab(train_x)
    vocab = find_vocab(dev_x,  vocab=vocab)
    output.write("Total vocab size: " + str(len(vocab))+ '\n')

    keys = initialize_vocab(vocab, params)

    dev_x = np.asarray(dev_x)
    dev_y = np.asarray(dev_y)
    train_x = np.asarray(train_x)
    train_y = np.asarray(train_y)

    output.write("train set size: " + str(len(train_y))+ ' examples, '
        + str(len(train_y)/params['BATCH_SIZE']+1) + ' batches per epoch\n')
    output.write("dev set size: " + str(len(dev_y))+ ' examples\n\n')
    # x encodes data: [batch size, l * word vector length]
    # y_ encodes labels: [batch size, classes]
    # x = tf.placeholder(tf.float32, [params['tensorflow_batch_size'], params['MAX_LENGTH'] * params['WORD_VECTOR_LENGTH']])
    # y_ = tf.placeholder(tf.float32, [params['tensorflow_batch_size'], params['CLASSES']])
    x = tf.placeholder(tf.float32, [None, params['MAX_LENGTH'] * params['WORD_VECTOR_LENGTH']])
    y_ = tf.placeholder(tf.float32, [None, params['CLASSES']])

    # print tf.shape(x)[0]
    # is x one tensor, or many??
    x = tf.reshape(x, [-1, params['MAX_LENGTH'], 1, params['WORD_VECTOR_LENGTH']])

    #init lists for convolutional layer
    slices = []
    weights = []
    biases = []
    #loop over KERNEL_SIZES, each time initializing a slice
    for kernel_size in params['KERNEL_SIZES']:
        slices, weights, biases = define_nn(x, kernel_size, params, slices, weights, biases)

    h_pool = tf.concat(len(params['KERNEL_SIZES']), slices)

    #apply dropout (p = TRAIN_DROPOUT or TEST_DROPOUT)
    dropout = tf.placeholder(tf.float32)
    h_pool_drop = tf.nn.dropout(h_pool, dropout)

    h_pool_flat = tf.squeeze(h_pool_drop)
    #fully connected softmax layer
    W_fc = weight_variable([len(params['KERNEL_SIZES']) * params['FILTERS'], params['CLASSES']])
    b_fc = bias_variable([params['CLASSES']])
    y_conv = tf.nn.softmax(tf.matmul(h_pool_flat, W_fc) + b_fc)

    #define error for training steps
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv),
                                   reduction_indices=[1]))
    if params['Adagrad']:
        train_step = tf.train.AdagradOptimizer(params['LEARNING_RATE']).minimize(cross_entropy)
    else:
        train_step = tf.train.AdamOptimizer(params['LEARNING_RATE']).minimize(cross_entropy)
    #define accuracy for evaluation
    correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    num_correct = tf.reduce_sum(tf.cast(correct_prediction, tf.float32))

    batches_x, batches_y = get_batches(train_size, params, train_x, train_y)

    #run session
    output.write( 'Initializing session...\n\n')
    sess = tf.Session(config=tf.ConfigProto(inter_op_parallelism_threads=2,
                      intra_op_parallelism_threads=3, use_per_session_threads=True))
    sess.run(tf.initialize_all_variables())
    output.write( 'Running session...\n\n')

    """
    train_accuracy = accuracy.eval(feed_dict={
        x: sub_vectors(train_x, keys, params), y_: train_y, dropout: 1.0}, session = sess)
    """
    best_dev_accuracy = 0
    output.write("initial accuracy %g \n"%accuracy.eval(feed_dict={x: sub_vectors(train_x, keys, params), y_: train_y, dropout: 1.0}, session = sess))
    for i in range(params['EPOCHS']):
        params['epoch'] = i + 1
        for j in range(batches_x.shape[0]/params['BATCH_SIZE']):
            batch_x, batch_y = get_batch(batches_x, batches_y, j, keys, params)
            params['tensorflow_batch_size'] = batch_y.shape[0]
            train_step.run(feed_dict={x: batch_x, y_: batch_y, dropout: params['TRAIN_DROPOUT']}, session = sess)
            #apply l2 clipping to weights and biases
            with sess.as_default():
                check_l2 = tf.reduce_sum(weights[0]).eval()
                for W in weights:
                    W = tf.clip_by_average_norm(W, params['L2_NORM_CONSTRAINT'])
                for b in biases:
                    b = tf.clip_by_average_norm(b, params['L2_NORM_CONSTRAINT'])
                W_fc = tf.clip_by_average_norm(W_fc, params['L2_NORM_CONSTRAINT'])
                b_fc = tf.clip_by_average_norm(b_fc, params['L2_NORM_CONSTRAINT'])
                if np.asscalar(check_l2) > np.asscalar(tf.reduce_sum(weights[0]).eval()):
                    output.write('weights clipped\n')
        batches_x, batches_y = shuffle_in_unison(batches_x, batches_y)
        train_softmax = cross_entropy.eval(feed_dict={x: sub_vectors(train_x, keys, params), y_: train_y, dropout: 1.0}, session = sess)
        output.write("epoch %d, training accuracy %g, training softmax error %g \n"
            %(i, accuracy.eval(feed_dict={x: sub_vectors(train_x, keys, params), y_: train_y, dropout: 1.0}, session = sess), train_softmax))

        # cross_entropy_accuracy = cross_entropy.eval(feed_dict={
        #     x: sub_vectors(train_x, keys, params), y_: train_y, dropout: 1.0}, session = sess)
        # print("epoch %d, softmax error %g"%(i, cross_entropy_accuracy))

        #prints accuracy for dev set every epoch, DEV is a hyperparameter boolean
        dev_softmax = cross_entropy.eval(feed_dict={x: sub_vectors(dev_x, keys, params), y_: dev_y, dropout: 1.0}, session = sess)
        dev_accuracy = accuracy.eval(feed_dict={x: sub_vectors(dev_x, keys, params), y_: dev_y, dropout: 1.0}, session = sess)
        output.write("dev set accuracy %g, softmax %g \n"%(dev_accuracy, dev_softmax))

        if dev_accuracy > best_dev_accuracy:
            #save model
            best_dev_accuracy = dev_accuracy
            """
        if dev_accuracy < best_dev_accuracy - .02:
            #early stop if accuracy drops significantly
            break"""

    output.close()
if __name__ == "__main__": main(sys.argv[1:])

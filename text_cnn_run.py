#feed_dict rewrites the value of tensors in the graph
# implement updating vocab
# initialize 1 minibatch at a time: store dev/train as text
# randomly init word vectors

import tensorflow as tf
import random
import linecache
from text_cnn_methods import *
import numpy as np

#define hyperparameters
def define_globals():
    params = {'WORD_VECTOR_LENGTH' : 300,
        'FILTERS' : 10,
        'KERNEL_SIZES' : [3],
        'CLASSES' : 2,
        'MAX_LENGTH' : 59,

        'L2_NORM_CONSTRAINT' : 3,
        'TRAIN_DROPOUT' : 0.5,

        'TRAINING_STEPS' : 20000,
        'BATCH_SIZE' : 50,
        'EPOCHS' : 50,
        'epoch' : 1,

        'rho' : 0.95,
        'epsilon' : 1e-6,
        'LEARNING_RATE' : 1e-7, #I think

        'TRAIN_FILE_NAME' : 'train-short',
        'DEV_FILE_NAME' : 'dev-eshort',
        'WORD_VECS_FILE_NAME' : 'output-test.txt',
        'SST' : True,
        'DEV' : False,

        'line_index' : 0,
        'batch_iterations' : 0,
        #debug
        'key_errors' : []
        }
    return params


#get random batch of examples from test file
def get_batches(lines, params, train_x, train_y):
    """
    batch_x = []
    batch_y = []

    np.random.seed(3435)
    if len(train_file_labels) % params['BATCH_SIZE'] > 0:
        for i in range(len(train_file_labels)/params['BATCH_SIZE']):
            for j in range(i * params['BATCH_SIZE'])
"""
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
"""

    if params['line_index'] + params['BATCH_SIZE'] <= lines:
        for line in range(params['line_index'], params['line_index'] + params['BATCH_SIZE']):
            batch_x.append(train_file_list[line])
            batch_y.append(train_file_labels[line])
        params['line_index'] += params['BATCH_SIZE']

    else:
        for line in range(params['line_index'], lines):
            print len(batch_x), len(train_file_list), line
            batch_x.append(train_file_list[line])
            batch_y.append(train_file_labels[line])
        for line in range(params['BATCH_SIZE'] - (lines - params['line_index'])):
            print len(batch_x), len(train_file_list), line
            batch_x.append(train_file_list[line])
            batch_y.append(train_file_labels[line])
        params['line_index'] = params['BATCH_SIZE'] - (lines - params['line_index'])
    return batch_x, batch_y
"""
#index and loop through same batches again
def get_batch(batches_x, batches_y, index, d, params):
    cur_batch_x = batches_x[index*params['BATCH_SIZE']:(index+1)*params['BATCH_SIZE'],:]
    cur_batch_y = batches_y[index*params['BATCH_SIZE']:(index+1)*params['BATCH_SIZE'],:]

    # print index
    """
    print batches_x.shape
    print batches_y.shape
    print type (batches_x.shape[0])
    batch_x = batches_x[batches_x.shape[0]-1,:]
    batch_y = batches_y[batches_y.shape[0]-1,:]
    batches_x = batches_x[:batches_x.shape[0]-1,:]
    batches_y = batches_y[:batches_y.shape[0]-1,:]
    """
    return sub_vectors(cur_batch_x, d, params), cur_batch_y

def main():
    params = define_globals()

    train_size = find_lines(params['TRAIN_FILE_NAME'] + '.labels')
    dev_size = find_lines(params['DEV_FILE_NAME'] + '.labels')

    #create training and eval sets
    dev_x, dev_y = get_all(params['DEV_FILE_NAME'], dev_size, params)
    train_x, train_y = get_all(params['TRAIN_FILE_NAME'], train_size, params)

    vocab = find_vocab(train_x)
    vocab = find_vocab(train_y,  vocab=vocab)
    keys = initialize_vocab(vocab, params['WORD_VECS_FILE_NAME'])

    # print list(keys.keys())
    dev_x = np.asarray(dev_x)
    dev_y = np.asarray(dev_y)
    train_x = np.asarray(train_x)
    train_y = np.asarray(train_y)
    #keys = {}
    #train_size = 500
    #dev_size = 500
    # x encodes data: [batch size, l * word vector length]
    # y_ encodes labels: [batch size, classes]
    x = tf.placeholder(tf.float32, [None, params['MAX_LENGTH'] * params['WORD_VECTOR_LENGTH']])
    y_ = tf.placeholder(tf.float32, [None, params['CLASSES']])

    # print tf.shape(x)[0]
    x = tf.reshape(x, [-1, params['MAX_LENGTH'], 1, params['WORD_VECTOR_LENGTH']])
    # print tf.shape(x)

    #init lists for convolutional layer if DNE
    try: slices
    except NameError: slices = []
    try: weights
    except NameError: weights = []
    try: biases
    except NameError: biases = []
    #loop over KERNEL_SIZES, each time initializing a slice
    for kernel_size in params['KERNEL_SIZES']:
        slices, weights, biases = define_nn(x, kernel_size, params, slices, weights, biases)
    # print slices
    h_pool = tf.concat(len(params['KERNEL_SIZES']), slices)
    # print h_pool
    #apply dropout (p = TRAIN_DROPOUT or TEST_DROPOUT)
    dropout = tf.placeholder(tf.float32)
    h_pool_drop = tf.nn.dropout(h_pool, dropout)

    h_pool_flat = tf.squeeze(h_pool_drop)
    #fully connected softmax layer
    W_fc = weight_variable([len(params['KERNEL_SIZES']) * params['FILTERS'], params['CLASSES']])
    b_fc = bias_variable([params['CLASSES']])
    y_conv = tf.nn.softmax(tf.matmul(h_pool_flat, W_fc) + b_fc)

    #define error for training steps
    #I'm pretty sure this is the same as reducing -log likelihood, which is what Kim uses
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv),
                                   reduction_indices=[1]))
    #train_step = tf.train.AdadeltaOptimizer(learning_rate = 1.3, rho = params['rho'], epsilon = params['epsilon']).minimize(cross_entropy)
    train_step = tf.train.GradientDescentOptimizer(.5).minimize(cross_entropy)
    #define accuracy for evaluation
    correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    batches_x, batches_y = get_batches(train_size, params, train_x, train_y)

    #run session
    print "Initializing session..."
    print ''
    sess = tf.Session(config=tf.ConfigProto(inter_op_parallelism_threads=1,
                      intra_op_parallelism_threads=10, use_per_session_threads=True))
    sess.run(tf.initialize_all_variables())
    print 'Running session...'
    print ''
    for i in range(params['EPOCHS']):
        params['epoch'] = i
        for j in range(batches_x.shape[0]/params['BATCH_SIZE']):
            batch_x, batch_y = get_batch(batches_x, batches_y, j, keys, params)
            #add code to create the word vector matrix for the words chosen
            train_step.run(feed_dict={x: batch_x, y_: batch_y, dropout: params['TRAIN_DROPOUT']}, session = sess)
            #train_step.run(feed_dict={x: batch_x, y_: batch_y}, session = sess)
            # if params['epoch'] < 5:
                # print batch_x[0]
                # print batch_y[0]
                #
                # print 'here!'
                # print weights[0].eval(session = sess)
            """
            #update weights
            for W in weights:
                W = l2_normalize(W, params['L2_NORM_CONSTRAINT'], sess)
            W_fc = l2_normalize(W_fc, params['L2_NORM_CONSTRAINT'], sess)
            #update biases
            for b in biases:
                b = l2_normalize(b, params['L2_NORM_CONSTRAINT'], sess)
            b_fc = l2_normalize(b_fc, params['L2_NORM_CONSTRAINT'], sess)"""
        train_accuracy = accuracy.eval(feed_dict={
            x: sub_vectors(train_x, keys, params), y_: train_y, dropout: 1.0}, session = sess)
        print("epoch %d, training accuracy %g"%(i, train_accuracy))
        #prints accuracy for dev set every 1000 examples, DEV is a hyperparameter boolean
        if params['DEV']:
            print("dev set accuracy %g"%accuracy.eval(feed_dict={
                x: dev_x,
                y_: dev_y,
                dropout: 1.0},
                session = sess))

    #print dev accuracy of results
    if params['DEV']:
        print("dev set accuracy %g"%accuracy.eval(feed_dict={x: dev_x, y_: dev_y, dropout: 1.0}, session = sess))

if __name__ == "__main__": main()

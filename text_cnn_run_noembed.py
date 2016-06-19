
#other notes:
    # Adam does got lower softmax, by a couple %
    # dev softmax min is before dev accuracy max -- typically after epoch 2-4 (zero-based)
    # should we save the model with lowest cross entropy or highest accuracy?
    # learning rate should be higher with random init, lower when we update word vecs, lower for larger datasets

#minor changes:
    #shuffled batches each epoch

#todo:
    # debug changes in accuracy--is it the embeddings??
    # test amazon, congress, convote
    # what causes programs to stop??
    # checkpoint code
    # test set
    # clean up--more methods

#program expects:
    # flags: -a for Adagrad, -u for updating, -w for use word2vec, -t for use tfidf
        # default = Adam, no updating, random init w/o tfidf
    # argv[1] directory with train.data, dev.data, train.labels, dev.labels in SST format
    # argv[2] learning rate
    # argv[3] number of epochs
    # argv[4] tfidf ('True' or 'False')
    # argv[5] identifier tag (appended to filename to distinguish multiple runs)

#outputs in file named
    # directory, Optimizer name, number of epochs, identifier .txt
    # (commas only where necessary to distinguish numbers)
        # initial accuracy (train data)
        # training and dev accuracy at each epoch, dev softmax accuracy
import numpy as np
import tensorflow as tf
import random
import linecache
from text_cnn_methods_noembed import *
import previous_text_cnn_methods
import sys, getopt
import os
import time
import logging
from sklearn.feature_extraction.text import TfidfVectorizer

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
        'MAX_EPOCH_SIZE' : 10000,

        'Adagrad' : False,
        'LEARNING_RATE' : args[1],
        'USE_TFIDF' : False,
        'USE_WORD2VEC' : False,
        'UPDATE_WORD_VECS' : False,
        'TRAIN_FILE_NAME' : 'train',
        'DEV_FILE_NAME' : 'dev',
        'WORD_VECS_FILE_NAME' : 'output.txt',
        'OUTPUT_FILE_NAME' : 'noembed' + args[0],
        'SST' : True,
        'ICMB' : False,
        'TREC' : False,
        'TEST' : False,

        #set by program-do not change!
        'epoch' : 1,
        'l2-loss' : tf.constant(0),
        #debug
        'key_errors' : [],
        'changes' : 0}
    return params

def analyze_argv(argv):
    try:
        opts, args = getopt.getopt(argv, "wtause")
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
    params = analyze_opts(opts, params)
    return args, params

def analyze_opts(opts, params):
    for opt in opts:
        if opt[0] == ("-a"):
            params['Adagrad'] = True
            params['OUTPUT_FILE_NAME'] += 'Adagrad'
            break
    if params['Adagrad'] == False:
        params['OUTPUT_FILE_NAME'] += 'Adam'
    params['OUTPUT_FILE_NAME'] += str(params['LEARNING_RATE'])
    for opt in opts:
        if opt[0] == ('-w'):
            params['USE_WORD2VEC'] = True
        if opt[0] == ('-t'):
            params['USE_TFIDF'] = True
        if opt[0] == ('-u'):
            params['UPDATE_WORD_VECS'] = True
        if opt[0] == ('-s'):
            params['BATCH_SIZE'] = 1
            params['OUTPUT_FILE_NAME'] += 'sgd'
        if opt[0] == ('-e'):
            params['TEST'] = True
    return params

def sum_prob(x, y_, bundle, params, correct_prediction, dropout, sess):
        all_x, all_y, incomplete, extras, examples_total = bundle
        sum_correct = tf.reduce_sum(tf.cast(correct_prediction, dtype=tf.float32))
        examples_correct = 0
        if incomplete == False:
            while len(all_x) > 0:
                examples_correct += sum_correct.eval(feed_dict={x: all_x[0],
                    y_: all_y[0], dropout: 1.0}, session = sess)
                all_x = all_x[1:]
                all_y = all_y[1:]
        else:
            while len(all_x) > 1:
                examples_correct += sum_correct.eval(feed_dict={x: all_x[0],
                    y_: all_y[0], dropout: 1.0}, session = sess)
                all_x = all_x[1:]
                all_y = all_y[1:]
            final_batch = np.asarray(correct_prediction.eval(feed_dict={x: all_x[0], y_: all_y[0], dropout: 1.0}, session = sess))
            for i in range(0, params['BATCH_SIZE'] - extras):
                if final_batch[i] == True:
                    examples_correct += 1
        return float(examples_correct) / examples_total

def define_nn(params):
    x = tf.placeholder(tf.int32, [params['BATCH_SIZE'], None])
    y_ = tf.placeholder(tf.float32, [params['BATCH_SIZE'], params['CLASSES']])

    # word_embeddings = tf.Variable(tf.convert_to_tensor(key_array, dtype = tf.float32),
    #                               trainable = params['UPDATE_WORD_VECS'])
    # embedding_layer = tf.nn.embedding_lookup(word_embeddings, x)
    # embedding_output = tf.reshape(embedding_layer,
    #                 [params['BATCH_SIZE'], -1, 1, params['WORD_VECTOR_LENGTH']])

    #init lists for convolutional layer
    slices = []
    weights = []
    biases = []
    #loop over KERNEL_SIZES, each time initializing a slice
    for kernel_size in params['KERNEL_SIZES']:
        slices, weights, biases = conv_slices(x, kernel_size,
                                            params, slices, weights, biases)
        # slices, weights, biases = conv_slices(embedding_output, kernel_size,
        #                                     params, slices, weights, biases)
    # output.write('debug' + str(slices[0]))
    h_pool = tf.concat(3, slices)
    #apply dropout (p = TRAIN_DROPOUT or TEST_DROPOUT)
    dropout = tf.placeholder(tf.float32)
    h_pool_drop = tf.nn.dropout(h_pool, dropout)

    h_pool_flat = tf.reshape(h_pool_drop, [params['BATCH_SIZE'], -1])
    # output.write('debug' + str(h_pool_flat))
    #fully connected softmax layer
    W_fc = weight_variable([len(params['KERNEL_SIZES']) * params['FILTERS'],
                            params['CLASSES']])
    b_fc = bias_variable([params['CLASSES']])
    y_conv = tf.nn.softmax(tf.matmul(h_pool_flat, W_fc) + b_fc)

    #define error for training steps
    log_loss = -tf.reduce_sum(y_ * tf.log(y_conv), reduction_indices=[1])
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv), reduction_indices=[1]))

    #define accuracy for evaluation
    correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
    return x, y_, dropout, weights, biases, W_fc, b_fc, log_loss, correct_prediction



def train(params, output, train_eval_bundle, dev_bundle, batches_x, batches_y, key_array, embed_keys, train_x, train_y):
    with tf.Graph().as_default():
        x, y_, dropout, weights, biases, W_fc, b_fc, log_loss, correct_prediction = define_nn(params)
        if params['Adagrad']:
            train_step = tf.train.AdagradOptimizer(params['LEARNING_RATE']).minimize(cross_entropy)
        else:
            train_step = tf.train.AdamOptimizer(params['LEARNING_RATE']).minimize(cross_entropy)

        saver = tf.train.Saver(tf.all_variables())
        #run session
        output.write( 'Initializing session...\n\n')
        sess = tf.Session(config=tf.ConfigProto(inter_op_parallelism_threads=2,
                          intra_op_parallelism_threads=3, use_per_session_threads=True))
        sess.run(tf.initialize_all_variables())
        output.write( 'Running session...\n\n')
        output.write('setup time: %g\n'%(time.clock()))
        best_dev_accuracy = 0
        train_softmax = sum_prob(x, y_, train_eval_bundle, params, log_loss, dropout, sess)
        initial_accuracy = sum_prob(x, y_, train_eval_bundle, params, correct_prediction, dropout, sess)
        output.write("initial accuracy %g softmax%g \n"%(initial_accuracy, train_softmax))
        output.write('start time: ' + str(time.clock()) + '\n')
        time_index = time.clock()
        epoch_time = 0
        for i in range(params['EPOCHS']):
            params['epoch'] = i + 1
            for j in range(len(batches_x)):
                train_step.run(feed_dict={x: batches_x[j],
                                          y_: batches_y[j],
                                          dropout: params['TRAIN_DROPOUT']},
                                          session = sess)
                #apply l2 clipping to weights and biases
                with sess.as_default():
                    # print weights[0].eval()
                    if j == 0:
                        l2_loss = tf.div(tf.sqrt(tf.nn.l2_loss(weights[0])), tf.convert_to_tensor(2.0)).eval()
                        output.write('l2 loss is %g' %l2_loss)
                    check_l2 = tf.reduce_sum(weights[0]).eval()
                    for W in weights:
                        W = tf.clip_by_average_norm(W, params['L2_NORM_CONSTRAINT'])
                    for b in biases:
                        b = tf.clip_by_average_norm(b, params['L2_NORM_CONSTRAINT'])
                    W_fc = tf.clip_by_average_norm(W_fc, params['L2_NORM_CONSTRAINT'])
                    b_fc = tf.clip_by_average_norm(b_fc, params['L2_NORM_CONSTRAINT'])
                    if np.asscalar(check_l2) > np.asscalar(tf.reduce_sum(weights[0]).eval()):
                        output.write('weights clipped\n')
            if params['BATCH_SIZE'] == 1:
                batches_x, batches_y = shuffle_in_unison(batches_x, batches_y)
            else:
                batches_x, batches_y = scramble_batches(train_x, train_y, params, embed_keys, train_eval_bundle[2], train_eval_bundle[3])
            train_softmax = sum_prob(x, y_, train_eval_bundle, params, log_loss, dropout, sess)

            train_accuracy = sum_prob(x, y_, train_eval_bundle, params, correct_prediction, dropout, sess)

            output.write("epoch %d, training accuracy %g, training softmax error %g \n"
                %(i, train_accuracy, train_softmax))

            dev_accuracy = sum_prob(x, y_, dev_bundle, params, correct_prediction, dropout, sess)
            dev_softmax = sum_prob(x, y_, dev_bundle, params, log_loss, dropout, sess)
            output.write("dev set accuracy %g, softmax %g \n"%(dev_accuracy, dev_softmax))

            if dev_accuracy > best_dev_accuracy:
                saver.save(sess, 'text_cnn_run' + params['OUTPUT_FILE_NAME'], global_step = params['epoch'])
                best_dev_accuracy = dev_accuracy

            if dev_accuracy < best_dev_accuracy - .02:
                #early stop if accuracy drops significantly
                break
            output.write('epoch time : ' + str(time.clock() - time_index))
            epoch_time += time.clock() - time_index
            time_index = time.clock()
            output.write('. elapsed: ' + str(time.clock()) + '\n')
        # if params['TEST']:
        #     output.write('Testing:\n')
        #     test_x, test_y = sort_examples_by_length(test_x, test_y)
        #     test_bundle = batch(test_x, test_y, params, embed_keys) + (len(test_y),)
        #     saver.restore
        #     test_accuracy = sum_prob(x, y_, test_bundle, params, correct_prediction, dropout, sess)
        #     output.write('Final test accuracy: %g' %test_accuracy)

        return epoch_time


def main(argv):
    args, params = analyze_argv(argv)

    output = initial_print_statements(params, args)
    sys.stderr = output
    train_x, train_y = get_all(args[0], params['TRAIN_FILE_NAME'], params)

    dev_x, dev_y = get_all(args[0], params['DEV_FILE_NAME'], params)
    if params['TEST']:
        test_x, test_y = get_all(args[0], 'test', params)
    else:
        test_x, test_y = [],[]
    params['MAX_LENGTH'] = get_max_length(train_x + dev_x + test_x)
    vocab = find_vocab(train_x + dev_x + test_x, params)
    keys = initialize_vocab(vocab, params)
    train_x, train_y = sort_examples_by_length(train_x, train_y)
    dev_x, dev_y = sort_examples_by_length(dev_x, dev_y)
    train_eval_bundle = batch(train_x, train_y, params, keys) + (len(train_y),)
    dev_bundle = batch(dev_x, dev_y, params, keys) + (len(dev_y),)
    # train_eval_bundle = batch(train_x, train_y, params, embed_keys) + (len(train_y),)
    # dev_bundle = batch(dev_x, dev_y, params, embed_keys) + (len(dev_y),)
    if params['BATCH_SIZE'] == 1:
        batches_x, batches_y = train_eval_bundle[:2]
    else:
        batches_x, batches_y = scramble_batches(train_x, train_y, params, keys, train_eval_bundle[2], train_eval_bundle[3])
        # batches_x, batches_y = scramble_batches(train_x, train_y, params, embed_keys, train_eval_bundle[2], train_eval_bundle[3])
    output.write("Total vocab size: " + str(len(vocab))+ '\n')
    output.write('train set size: %d examples, %d batches per epoch\n'%(len(train_y), len(train_eval_bundle[0])))
    output.write("dev set size: " + str(len(dev_y))+ ' examples\n\n')
    epoch_time = train(params, output, train_eval_bundle, dev_bundle, batches_x, batches_y, key_array, keys, train_x, train_y)
    output.write('avg time: ' + str(epoch_time/params['EPOCHS']))
    sys.stderr.close()
    sys.stderr = sys.__stderr__
    output.close()
if __name__ == "__main__": main(sys.argv[1:])

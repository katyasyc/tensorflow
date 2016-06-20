
#other notes:
    # Adam does got lower softmax, by a couple %
    # dev softmax min is before dev accuracy max -- typically after epoch 2-4 (zero-based)
    # should we save the model with lowest cross entropy or highest accuracy?
    # learning rate should be higher with random init, lower when we update word vecs, lower for larger datasets

#minor changes:
    #shuffled batches each epoch

#todo:
    # debug changes in accuracy--is it the embeddings??
    # test with random addition of padding
    # what causes programs to stop??
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
from text_cnn_methods import *
import previous_text_cnn_methods
import sys, argparse
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
        'EPOCHS' : args.EPOCHS,
        'MAX_EPOCH_SIZE' : 10000,

        'Adagrad' : args.Adagrad,
        'LEARNING_RATE' : args.LEARNING_RATE,
        'USE_TFIDF' : args.tfidf,
        'USE_WORD2VEC' : args.word2vec,
        'USE_DELTA' : args.delta,
        'FLEX' : args.flex,
        'UPDATE_WORD_VECS' : args.update,
        'DIR' : args.path,
        'TRAIN_FILE_NAME' : 'train',
        'DEV_FILE_NAME' : 'dev',
        'WORD_VECS_FILE_NAME' : 'output.txt',
        'OUTPUT_FILE_NAME' : '' + str(args.path),
        'SST' : False,
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

def embed_layer(x, params, key_array, word_embeddings):
    if params['USE_DELTA']:
        W_delta = tf.Variable(tf.constant(1.0, shape=key_array.shape))
        weighted_word_embeddings = tf.mul(word_embeddings, W_delta)
        embedding_layer = tf.nn.embedding_lookup(weighted_word_embeddings, x)
    else:
        embedding_layer = tf.nn.embedding_lookup(word_embeddings, x)
    return tf.reshape(embedding_layer,
                    [params['BATCH_SIZE'], -1, 1, params['WORD_VECTOR_LENGTH']])

def fully_connected_layer(slices, params):
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
    return tf.nn.softmax(tf.matmul(h_pool_flat, W_fc) + b_fc), dropout, W_fc, b_fc

def define_nn(params, key_array):
    x = tf.placeholder(tf.int32, [params['BATCH_SIZE'], None])
    y_ = tf.placeholder(tf.float32, [params['BATCH_SIZE'], params['CLASSES']])
    word_embeddings = tf.Variable(tf.convert_to_tensor(key_array, dtype = tf.float32),
                                  trainable = params['UPDATE_WORD_VECS'])
    embedding_output = embed_layer(x, params, key_array, word_embeddings)
    #init lists for convolutional layer
    slices = []
    weights = []
    biases = []
    #loop over KERNEL_SIZES, each time initializing a slice
    for kernel_size in params['KERNEL_SIZES']:
        slices, weights, biases = conv_slices(embedding_output, kernel_size,
                                            params, slices, weights, biases)
    y_conv, dropout, W_fc, b_fc = fully_connected_layer(slices, params)
    #define error for training steps
    log_loss = -tf.reduce_sum(y_ * tf.log(y_conv), reduction_indices=[1])

    #define accuracy for evaluation
    correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv), reduction_indices=[1]))
    return x, y_, dropout, weights, biases, W_fc, b_fc, log_loss, correct_prediction, cross_entropy

def get_batches(params, train_eval_bundle, batches_bundle):
    if params['BATCH_SIZE'] == 1:
        return train_eval_bundle[:2]
    else:
        return scramble_batches(params, train_eval_bundle, batches_bundle)

def regularize(output, weights, W_fc, biases, b_fc, params, sess):
    with sess.as_default():
        # if j == 0:
        #     l2_loss = tf.div(tf.sqrt(tf.nn.l2_loss(weights[0])), tf.convert_to_tensor(2.0)).eval()
        #     output.write('l2 loss is %g\n' %l2_loss)
        check_l2 = tf.reduce_sum(weights[0]).eval()
        for W in weights:
            W = tf.clip_by_average_norm(W, params['L2_NORM_CONSTRAINT'])
        for b in biases:
            b = tf.clip_by_average_norm(b, params['L2_NORM_CONSTRAINT'])
        W_fc = tf.clip_by_average_norm(W_fc, params['L2_NORM_CONSTRAINT'])
        b_fc = tf.clip_by_average_norm(b_fc, params['L2_NORM_CONSTRAINT'])
        if np.asscalar(check_l2) > np.asscalar(tf.reduce_sum(weights[0]).eval()):
            output.write('weights clipped\n')
    return weights, W_fc, biases, b_fc

def print_eval(output, name, x, y_, bundle, params, log_loss, correct_prediction, dropout, sess):
    softmax = sum_prob(x, y_, bundle, params, log_loss, dropout, sess)
    accuracy = sum_prob(x, y_, bundle, params, correct_prediction, dropout, sess)
    output.write(name + ' accuracy %g softmax %g \n'%(accuracy, softmax))
    return accuracy

def train(params, output, data):
    train_eval_bundle, dev_bundle, test_bundle, batches_bundle, key_array = data
    with tf.Graph().as_default():
        x, y_, dropout, weights, biases, W_fc, b_fc, log_loss, correct_prediction, cross_entropy = define_nn(params, key_array)
        if params['Adagrad']:
            train_step = tf.train.AdagradOptimizer(params['LEARNING_RATE']).minimize(cross_entropy)
        else:
            train_step = tf.train.AdamOptimizer(params['LEARNING_RATE']).minimize(cross_entropy)
        batches_x, batches_y = get_batches(params, train_eval_bundle, batches_bundle)
        saver = tf.train.Saver(tf.all_variables())
        #run session
        output.write( 'Initializing session...\n\n')
        sess = tf.Session(config=tf.ConfigProto(inter_op_parallelism_threads=2,
                          intra_op_parallelism_threads=3, use_per_session_threads=True))
        sess.run(tf.initialize_all_variables())
        output.write( 'Running session...\n\n')
        output.write('setup time: %g\n'%(time.clock()))
        best_dev_accuracy = 0
        print_eval(output, 'initial', x, y_, train_eval_bundle, params, log_loss, correct_prediction, dropout, sess)
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
                weights, W_fc, biases, b_fc = regularize(output, weights, W_fc, biases, b_fc, params, sess)
                #end do weights method
                batches_x, batches_y = get_batches(params, train_eval_bundle, batches_bundle)
            output.write("epoch %d"%params['epoch'])
            print_eval(output, 'training', x, y_, train_eval_bundle, params, log_loss, correct_prediction, dropout, sess)

            dev_accuracy = print_eval(output, 'dev', x, y_, dev_bundle, params, log_loss, correct_prediction, dropout, sess)

            if dev_accuracy > best_dev_accuracy:
                if params['TEST']:
                    checkpoint = saver.save(sess, 'text_cnn_run' + params['OUTPUT_FILE_NAME'], global_step = params['epoch'])
                best_dev_accuracy = dev_accuracy

            # if dev_accuracy < best_dev_accuracy - .02:
            #     #early stop if accuracy drops significantly
            #     break
            output.write('epoch time : ' + str(time.clock() - time_index))
            epoch_time += time.clock() - time_index
            time_index = time.clock()
            output.write('. elapsed: ' + str(time.clock()) + '\n')
        output.write('Max accuracy ' + str(best_dev_accuracy) + '\n')
        if params['TEST']:
            output.write('Testing:\n')
            saver.restore(sess, checkpoint)
            test_accuracy = sum_prob(x, y_, test_bundle, params, correct_prediction, dropout, sess)
            output.write('Final test accuracy:')
        return epoch_time


def analyze_opts(args, params):
    if args.path == 'sst1':
        params['CLASSES'] = 5
        params['SST'] = True
    elif args.path == 'sst1':
        params['SST'] == True

    if params['Adagrad'] == True:
        params['OUTPUT_FILE_NAME'] += 'Adagrad'
    else:
        params['OUTPUT_FILE_NAME'] += 'Adam'
    params['OUTPUT_FILE_NAME'] += str(params['LEARNING_RATE'])

    if args.abbrev == True:
        params['KERNEL_SIZES'] = [3]
        params['FILTERS'] = 5
    if args.sgd == True:
        params['BATCH_SIZE'] = 1
        params['OUTPUT_FILE_NAME'] += 'sgd'
    return params

def get_data(params, output):
    train_x, train_y = get_all(params['DIR'], params['TRAIN_FILE_NAME'], params)

    dev_x, dev_y = get_all(params['DIR'], params['DEV_FILE_NAME'], params)
    if params['TEST']:
        test_x, test_y = get_all(params['DIR'], 'test', params)
    else:
        test_x, test_y = [],[]

    params['MAX_LENGTH'] = get_max_length(train_x + dev_x + test_x)
    vocab = find_vocab(train_x + dev_x + test_x, params)
    embed_keys, key_array = initialize_vocab(vocab, params)
    train_x, train_y = sort_examples_by_length(train_x, train_y)
    dev_x, dev_y = sort_examples_by_length(dev_x, dev_y)
    if params['TEST']:
        test_x, test_y = sort_examples_by_length(test_x, test_y)
        test_bundle = batch(test_x, test_y, params, embed_keys) + (len(test_y),)
    else:
        test_bundle = ()
    train_eval_bundle = batch(train_x, train_y, params, embed_keys) + (len(train_y),)
    dev_bundle = batch(dev_x, dev_y, params, embed_keys) + (len(dev_y),)
    batches_bundle = train_x, train_y, embed_keys
    output.write("Total vocab size: " + str(len(vocab))+ '\n')
    output.write('train set size: %d examples, %d batches per epoch\n'%(len(train_y), len(train_eval_bundle[0])))
    output.write("dev set size: " + str(len(dev_y))+ ' examples\n\n')
    return train_eval_bundle, dev_bundle, test_bundle, batches_bundle, key_array


def main(params, output):
    sys.stderr = output
    #make method
    data = get_data(params, output)
    #end method
    epoch_time = train(params, output, data)
    output.write('avg time: ' + str(epoch_time/params['EPOCHS']))
    sys.stderr.close()
    sys.stderr = sys.__stderr__
    output.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-a', '--Adagrad', action='store_true', default=False)
    parser.add_argument('-w', '--word2vec', action='store_true', default=False)
    parser.add_argument('-u', '--update', action='store_true', default=False)
    parser.add_argument('-d', '--delta', action='store_true', default=False)
    parser.add_argument('-e', '--test', action='store_true', default=False)
    parser.add_argument('-b', '--abbrev', action='store_true', default=False)
    parser.add_argument('-s', '--sgd', action='store_true', default=False)
    parser.add_argument('-t', '--tfidf', action='store_true', default=False)
    parser.add_argument('-f', '--flex', type=int, choices=xrange(1,10))
    parser.add_argument('path', type=str)
    parser.add_argument('LEARNING_RATE', type=float)
    parser.add_argument('EPOCHS', type=int)
    parser.add_argument('string')
    args = parser.parse_args()
    params = define_globals(args)
    params = analyze_opts(args, params)
    output = initial_print_statements(params, args)
    logging.basicConfig(filename=params['OUTPUT_FILE_NAME'], level=logging.DEBUG)
    try: main(params, output)
    except BaseException:
        logging.getLogger(__name__).exception("Program terminated")
        raise

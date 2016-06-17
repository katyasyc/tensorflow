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
    # should we save the model with lowest cross entropy or highest accuracy?

#minor changes:
#shuffled batches each epoch

#todo:
    # test amazon, congress, convote

    #possible issues:
    #weights not updating
    #accuracy not printing correctly


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
import sys, getopt
import os
import time
from sklearn.feature_extraction.text import TfidfVectorizer

#define hyperparameters
def define_globals(args):
    params = {'WORD_VECTOR_LENGTH' : 100,
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
        'OUTPUT_FILE_NAME' : args[0],
        'SST' : True,
        'ICMB' : False,
        'TREC' : False,

        #set by program-do not change!
        'epoch' : 1,
        'l2-loss' : tf.constant(0),
        #debug
        'key_errors' : [],
        'changes' : 0}
    return params

def analyze_argv(argv):
    try:
        opts, args = getopt.getopt(argv, "wtaus")
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
        if opt[0] == ('-w'):
            params['USE_WORD2VEC'] = True
        if opt[0] == ('-t'):
            params['USE_TFIDF'] = True
        if opt[0] == ('-u'):
            params['UPDATE_WORD_VECS'] = True
        if opt[0] == ('-s'):
            params['BATCH_SIZE'] = 1
            params['OUTPUT_FILE_NAME'] += 'sgd'
    return args, params

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

def main(argv):
    args, params = analyze_argv(argv)

    output = initial_print_statements(params, args)
    train_x, train_y = get_all(args[0], params['TRAIN_FILE_NAME'], params)

    dev_x, dev_y = get_all(args[0], params['DEV_FILE_NAME'], params)
    vocab = find_vocab(train_x + dev_x, params)

    embed_keys, key_list = initialize_vocab(vocab, params)
    train_x, train_y = sort_examples_by_length(train_x, train_y)
    dev_x, dev_y = sort_examples_by_length(dev_x, dev_y)
    train_eval_bundle = batch(train_x, train_y, params, embed_keys) + (len(train_y),)
    dev_bundle = batch(dev_x, dev_y, params, embed_keys) + (len(dev_y),)
    if params['BATCH_SIZE'] == 1:
        batches_x, batches_y = train_eval_bundle[:2]
    else:
        batches_x, batches_y = scramble_batches(train_x, train_y, params, embed_keys, train_eval_bundle[2], train_eval_bundle[3])

    #note: normalizing over average_weight improves accuracy by 2 points in trials
    #average_weight is 8 point something (!), which overshadows words in dev but not train
    if params['USE_TFIDF'] == True:
        train_x_str = get_strings(args[0], params['TRAIN_FILE_NAME'], params)
        idf_vectorizer = TfidfVectorizer(vocabulary=vocab).fit(train_x_str)
        average_weight = np.mean(idf_vectorizer.idf_)
        idf_array = np.stack((np.asarray(vocab), idf_vectorizer.idf_), 1)
        for i in range(idf_array.shape[0]):
            key_list[embed_keys[idf_array[i][0]]] = np.divide(np.multiply(
                                        float(idf_array[i][1]),
                                        key_list[embed_keys[idf_array[i][0]]]),
                                        average_weight)

    key_array = np.asarray(key_list)
    output.write("Total vocab size: " + str(len(vocab))+ '\n')
    output.write('train set size: %d examples, %d batches per epoch\n'%(len(train_y), len(train_eval_bundle[0])))
    output.write("dev set size: " + str(len(dev_y))+ ' examples\n\n')

    x = tf.placeholder(tf.int32, [params['BATCH_SIZE'], None])
    y_ = tf.placeholder(tf.float32, [params['BATCH_SIZE'], params['CLASSES']])

    word_embeddings = tf.Variable(tf.convert_to_tensor(key_array, dtype = tf.float32),
                                  trainable = params['UPDATE_WORD_VECS'])
    embedding_layer = tf.nn.embedding_lookup(word_embeddings, x)
    embedding_output = tf.reshape(embedding_layer,
                    [params['BATCH_SIZE'], -1, 1, params['WORD_VECTOR_LENGTH']])

    #init lists for convolutional layer
    slices = []
    weights = []
    biases = []
    #loop over KERNEL_SIZES, each time initializing a slice
    for kernel_size in params['KERNEL_SIZES']:
        slices, weights, biases = define_nn(embedding_output, kernel_size,
                                            params, slices, weights, biases)
    # print slices
    h_pool = tf.concat(3, slices)
    # print h_pool
    #apply dropout (p = TRAIN_DROPOUT or TEST_DROPOUT)
    dropout = tf.placeholder(tf.float32)
    h_pool_drop = tf.nn.dropout(h_pool, dropout)

    h_pool_flat = tf.reshape(h_pool_drop, [params['BATCH_SIZE'], -1])
    # print h_pool_flat
    #fully connected softmax layer
    W_fc = weight_variable([len(params['KERNEL_SIZES']) * params['FILTERS'],
                            params['CLASSES']])
    b_fc = bias_variable([params['CLASSES']])
    y_conv = tf.nn.softmax(tf.matmul(h_pool_flat, W_fc) + b_fc)

    #define error for training steps
    log_loss = -tf.reduce_sum(y_ * tf.log(y_conv), reduction_indices=[1])
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv), reduction_indices=[1]))
    if params['Adagrad']:
        train_step = tf.train.AdagradOptimizer(params['LEARNING_RATE']).minimize(cross_entropy)
    else:
        train_step = tf.train.AdamOptimizer(params['LEARNING_RATE']).minimize(cross_entropy)
    #define accuracy for evaluation
    correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

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
        # train_softmax = cross_entropy.eval(feed_dict=
        #     {x: sub_indices(train_x, embed_keys),
        #     y_: train_y, dropout: 1.0}, session = sess)
        # print 'batches', batches_x[0].shape,batches_x[0], batches_y[0]
        # print 'train', train_eval_bundle[0][0].shape, train_eval_bundle[0][0], train_eval_bundle[1][0]
        # print 'dev', dev_bundle[0][0].shape, dev_bundle[0][0], dev_bundle[1][0]
        # print accuracy.eval(feed_dict={x: batches_x[0],
        #                           y_: batches_y[0],
        #                           dropout: params['TRAIN_DROPOUT']},
        #                           session = sess)
        train_softmax = sum_prob(x, y_, train_eval_bundle, params, log_loss, dropout, sess)

        train_accuracy = sum_prob(x, y_, train_eval_bundle, params, log_loss, dropout, sess)

        output.write("epoch %d, training accuracy %g, training softmax error %g \n"
            %(i, train_accuracy, train_softmax))

        # dev_softmax = cross_entropy.eval(feed_dict=
        #                             {x: sub_indices(dev_x, embed_keys),
        #                              y_: dev_y, dropout: 1.0}, session = sess)

        #does not evaluate correctly in method: variables don't  update
        all_x, all_y, incomplete, extras, examples_total = dev_bundle
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
                for k in range(0, params['BATCH_SIZE'] - extras):
                    if final_batch[k] == True:
                        examples_correct += 1
        dev_accuracy = sum_prob(x, y_, dev_bundle, params, correct_prediction, dropout, sess)
        dev_softmax = sum_prob(x, y_, dev_bundle, params, log_loss, dropout, sess)
        output.write("dev set accuracy %g, softmax %g \n"%(dev_accuracy, dev_softmax))

        if dev_accuracy > best_dev_accuracy:
            #save model
            best_dev_accuracy = dev_accuracy

        # if dev_accuracy < best_dev_accuracy - .02:
        #     #early stop if accuracy drops significantly
        #     break
        output.write('epoch time : ' + str(time.clock() - time_index))
        epoch_time += time.clock() - time_index
        time_index = time.clock()
        output.write('. elapsed: ' + str(time.clock()) + '\n')
    output.write('avg time: ' + str(epoch_time/params['EPOCHS']))
    output.close()
if __name__ == "__main__": main(sys.argv[1:])

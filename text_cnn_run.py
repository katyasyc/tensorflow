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
    # test updating weights
#fix memory
#time
#minibatch size
#fix minibatch size
#speed up by replacing words with indices before training

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
#beware:may break if first example has maximum length
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
    params = {'WORD_VECTOR_LENGTH' : 300,
        'FILTERS' : 10,
        'KERNEL_SIZES' : [3],
        'CLASSES' : 2,
        'MAX_LENGTH' : 59,

        'L2_NORM_CONSTRAINT' : 3.0,
        'TRAIN_DROPOUT' : 0.5,

        'BATCH_SIZE' : 50,
        'EPOCHS' : args[2],

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
        'DEV' : True,

        #set by program-do not change!
        'batch_iterations' : 0,
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
        #fix!!!
        if opt[0] == ('-s'):
            params['BATCH_SIZE'] = 2
    params['OUTPUT_FILE_NAME'] += 'sgd'
    return args, params

def sub_indices(input_x, embed_keys):
    example_list = []
    for sentence in input_x:
        example_indices = []
        for token in sentence:
            example_indices.append(embed_keys[token])
        example_list.append(example_indices)
    return np.asarray(example_list)

def sum_prob(x, y_,set_x, set_y, keys, params, correct_prediction, dropout, sess):
        set_x_temp = set_x
        set_y_temp = set_y
        sum_correct = tf.reduce_sum(tf.cast(correct_prediction, dtype=tf.float32))
        examples_correct = 0
        while set_y_temp.shape[0] >= params['BATCH_SIZE']:
            examples_correct += sum_correct.eval(feed_dict={x: set_x_temp[0:50], y_: set_y_temp[0:50], dropout: 1.0}, session = sess)
            set_x_temp = set_x_temp[50:]
            set_y_temp = set_y_temp[50:]
        if set_y_temp.shape[0] > 0:
            remaining_examples = set_y_temp.shape[0]
            set_y_temp = np.concatenate((set_y_temp, np.zeros((params['BATCH_SIZE'] - remaining_examples, params['CLASSES']))), axis = 0)
            zeroes = np.full((params['BATCH_SIZE']-remaining_examples, params['MAX_LENGTH']), keys['<PAD>'], dtype=int)
            set_x_temp = np.concatenate((set_x_temp, zeroes), axis = 0)
            final_batch = np.asarray(correct_prediction.eval(feed_dict={x: set_x_temp, y_: set_y_temp, dropout: 1.0}, session = sess))
            for i in range(0, remaining_examples):
                if final_batch[i] == True:
                    examples_correct += 1
        return float(examples_correct) / set_y.shape[0]


def main(argv):
    args, params = analyze_argv(argv)

    output = initial_print_statements(params, args)
    #possibly buggy??
    # params['MAX_LENGTH'] = get_max_length(args[0], params['TRAIN_FILE_NAME'], params['DEV_FILE_NAME'])
    train_x, train_y = get_all(args[0], params['TRAIN_FILE_NAME'], params)
    dev_x, dev_y = get_all(args[0], params['DEV_FILE_NAME'], params)
    vocab = find_vocab(train_x + dev_x)
    #xes are still one dim python lists w/sentences
    embed_keys, key_list = initialize_vocab(vocab, params)

    #note: normalizing over average_weight improves accuracy by 2 points in trials
    #average_weight is 8 point something (!), which overshadows words in dev but not train
    if params['USE_TFIDF'] == True:
        idf_vectorizer = TfidfVectorizer(vocabulary=vocab).fit(train_x)
        average_weight = np.mean(idf_vectorizer.idf_)
        idf_array = np.stack((np.asarray(vocab), idf_vectorizer.idf_), 1)
        for i in range(idf_array.shape[0]):
            key_list[embed_keys[idf_array[i][0]]] = np.divide(np.multiply(
                                        float(idf_array[i][1]),
                                        key_list[embed_keys[idf_array[i][0]]]),
                                        average_weight)

    dev_x = sub_indices(dev_x, embed_keys)
    train_x = sub_indices(train_x, embed_keys)
    dev_y = np.asarray(dev_y)
    train_y = np.asarray(train_y)

    output.write("Total vocab size: " + str(len(vocab))+ '\n')

    output.write("train set size: " + str(train_y.shape[0]) + ' examples, '
        + str(len(train_y)/params['BATCH_SIZE']+1) + ' batches per epoch\n')
    output.write("dev set size: " + str(dev_y.shape[0])+ ' examples\n\n')

    x = tf.placeholder(tf.int32, [None, params['MAX_LENGTH']])
    y_ = tf.placeholder(tf.float32, [None, params['CLASSES']])

    word_embeddings = tf.Variable(np.asarray(key_list),
                                  trainable = params['UPDATE_WORD_VECS'],
                                  dtype = tf.float32)
    embedding_layer = tf.nn.embedding_lookup(word_embeddings, x)
    embedding_output = tf.reshape(embedding_layer,
                    [-1, params['MAX_LENGTH'], 1, params['WORD_VECTOR_LENGTH']])

    #init lists for convolutional layer
    slices = []
    weights = []
    biases = []
    #loop over KERNEL_SIZES, each time initializing a slice
    for kernel_size in params['KERNEL_SIZES']:
        slices, weights, biases = define_nn(embedding_output, kernel_size,
                                            params, slices, weights, biases)
    h_pool = tf.concat(len(params['KERNEL_SIZES']), slices)

    #apply dropout (p = TRAIN_DROPOUT or TEST_DROPOUT)
    dropout = tf.placeholder(tf.float32)
    h_pool_drop = tf.nn.dropout(h_pool, dropout)

    h_pool_flat = tf.squeeze(h_pool_drop)
    #fully connected softmax layer
    W_fc = weight_variable([len(params['KERNEL_SIZES']) * params['FILTERS'],
                            params['CLASSES']])
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

    batches_x, batches_y = get_batches(params, train_x, train_y)

    #run session
    output.write( 'Initializing session...\n\n')
    sess = tf.Session(config=tf.ConfigProto(inter_op_parallelism_threads=2,
                      intra_op_parallelism_threads=3, use_per_session_threads=True))
    sess.run(tf.initialize_all_variables())
    output.write( 'Running session...\n\n')
    output.write('setup time: ' + str(time.clock()) + '\n')
    best_dev_accuracy = 0
    print sum_prob(x, y_, train_x, train_y, embed_keys, params, correct_prediction, dropout, sess)
    print accuracy.eval(feed_dict={x: train_x, y_: train_y, dropout: 1.0},session = sess)
    output.write("initial accuracy %g \n"%accuracy.eval(feed_dict=
        {x: train_x, y_: train_y, dropout: 1.0},
        session = sess))
    output.write('start time: ' + str(time.clock()) + '\n')
    time_counter = time.clock()
    epoch_time = 0
    for i in range(params['EPOCHS']):
        params['epoch'] = i + 1
        for j in range(batches_x.shape[0]/params['BATCH_SIZE']):
            batch_x, batch_y = get_batch(batches_x, batches_y, j, params)
            params['tensorflow_batch_size'] = batch_y.shape[0]
            train_step.run(feed_dict={x: batch_x,
                                      y_: batch_y,
                                      dropout: params['TRAIN_DROPOUT']},
                                      session = sess)
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
        train_softmax = cross_entropy.eval(feed_dict=
            {x: train_x,
            y_: train_y, dropout: 1.0}, session = sess)
        print sum_prob(x, y_, train_x, train_y, embed_keys, params, correct_prediction, dropout, sess)
        train_accuracy = accuracy.eval(feed_dict=
                                    {x: train_x,
                                     y_: train_y, dropout: 1.0}, session = sess)
        print train_accuracy
        output.write("epoch %d, training accuracy %g, training softmax error %g \n"
            %(i, train_accuracy, train_softmax))

        dev_softmax = cross_entropy.eval(feed_dict=
                                    {x: dev_x,
                                     y_: dev_y, dropout: 1.0}, session = sess)
        print sum_prob(x, y_, dev_x, dev_y, embed_keys, params, correct_prediction, dropout, sess)
        dev_accuracy = accuracy.eval(feed_dict=
                                    {x: dev_x,
                                     y_: dev_y, dropout: 1.0}, session = sess)

        output.write("dev set accuracy %g, softmax %g \n"%(dev_accuracy, dev_softmax))
        print dev_accuracy
        if dev_accuracy > best_dev_accuracy:
            #save model
            best_dev_accuracy = dev_accuracy

        # if dev_accuracy < best_dev_accuracy - .02:
        #     #early stop if accuracy drops significantly
        #     break
        output.write('epoch time : ' + str(time.clock() - time_counter))
        epoch_time += (time.clock()-time_counter)
        time_counter = time.clock()
        output.write('. elapsed: ' + str(time.clock()) + '\n')
    output.write('avg time: ' + str(epoch_time/params['EPOCHS']))
    output.close()
if __name__ == "__main__": main(sys.argv[1:])

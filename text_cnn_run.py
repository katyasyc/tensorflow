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
        opts, args = getopt.getopt(argv, "wtau")
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
    return args, params

def initial_print_statements(params, args):
    params['OUTPUT_FILE_NAME'] += ',' + str(params['EPOCHS'])
    if params['USE_TFIDF']:
        params['OUTPUT_FILE_NAME'] += 'tfidf'
    params['OUTPUT_FILE_NAME'] += ','
    if params['USE_WORD2VEC']:
        params['OUTPUT_FILE_NAME'] += 'word2vec'
    else:
        params['OUTPUT_FILE_NAME'] += 'randinit'
    params['OUTPUT_FILE_NAME'] += ','
    if params['UPDATE_WORD_VECS']:
        params['OUTPUT_FILE_NAME'] += 'upd'
    params['OUTPUT_FILE_NAME'] += args[3]
    output = open(params['OUTPUT_FILE_NAME'] + '.txt', 'a', 0)
    if params['Adagrad']:
        output.write("Running Adagrad on " + args[0] + " with a learning rate of ")
    else:
        output.write("Running Adam on " + args[0] + " with a learning rate of ")
    output.write(str(params['LEARNING_RATE']) + ' and ' + str(params['EPOCHS']) + ' epochs\n')
    output.write('Using ')
    if params['USE_TFIDF']:
        output.write('tfidf, ')
    else:
        output.write('no frequency weighting, ')
    if params['USE_WORD2VEC']:
        output.write('word2vec, ')
    else:
        output.write('random init, ')
    if params['UPDATE_WORD_VECS']:
        output.write('updating.\n')
    else:
        output.write('not updating.\n')
    return output

#convert data for feeding into placeholders
def feed(input_tfidf, input_x, keys, params):
    if params['USE_TFIDF'] == False:
        return np.expand_dims(sub_vectors(input_x, keys, params), 2)
    elif params['USE_WORD2VEC'] == True:
        vecs = sub_vectors(input_x, keys, params)
        for example_index in range(vecs.shape[0]):
            for word_index in range(params['MAX_LENGTH']):
                scalar = input_tfidf[example_index][word_index]
                for number_index in range(params['WORD_VECTOR_LENGTH']):
                    vecs[example_index][word_index][number_index] = scalar * vecs[example_index][word_index][number_index]
        return np.expand_dims(vecs, 2)
    else:
        indices = []
        values = []
        print "shape input_x:", input_x.shape[1]
        for sentence_index in range(input_x.shape[0]):
            for word_index in range(input_x.shape[1]):
                print word_index
                indices.append([sentence_index, word_index, input_x[sentence_index][word_index][0]])
                values.append(input_x[sentence_index][word_index][1])
                sys.stdout.flush()
        shape = np.array([input_x.shape[0], input_x.shape[1], len(vocab)])
        return tf.SparseTensorValue(np.asarray(indices), np.asarray(values), shape)

def sub_indices(input_x, embed_keys):
    example_list = []
    for sentence in input_x:
        example_indices = []
        for token in sentence:
            example_indices.append(embed_keys[token])
        example_list.append(example_indices)
    return np.asarray(example_list)

def flat_vectorize(input_x, input_x_str, vocab, params):
    vectorizer = TfidfVectorizer(vocabulary = vocab)
    input_vector = input_x.flatten().tolist()
    input_tfidf_sparse = coo_matrix(vectorizer.fit_transform(input_vector))
    input_tfidf_maxes = input_tfidf_sparse.max(1).toarray()
    return input_tfidf_maxes.reshape((input_x.shape[0], params['MAX_LENGTH']))

def main(argv):
    args, params = analyze_argv(argv)

    output = initial_print_statements(params, args)

    train_x, train_y = get_all(args[0], params['TRAIN_FILE_NAME'], params)
    dev_x, dev_y = get_all(args[0], params['DEV_FILE_NAME'], params)
    vocab = find_vocab(train_x + dev_x)

    embed_keys, key_list = initialize_vocab(vocab, params)

    if params['USE_TFIDF'] == True:
        train_x_str = get_strings(args[0], params['TRAIN_FILE_NAME'], params)
        idf_vectorizer = TfidfVectorizer(vocabulary=vocab).fit(train_x_str)
        average_weight = np.mean(idf_vectorizer.idf_)
        idf_array = np.stack((np.asarray(vocab), idf_vectorizer.idf_), 1)
        for i in range(idf_array.shape[0]):
            key_list[embed_keys[idf_array[i][0]]] = np.multiply(float(idf_array[i][1]), key_list[embed_keys[idf_array[i][0]]])

    key_array = np.asarray(key_list)
    dev_x = np.asarray(dev_x)
    train_x = np.asarray(train_x)
    dev_y = np.asarray(dev_y)
    train_y = np.asarray(train_y)

    output.write("Total vocab size: " + str(len(vocab))+ '\n')

    output.write("train set size: " + str(len(train_y))+ ' examples, '
        + str(len(train_y)/params['BATCH_SIZE']+1) + ' batches per epoch\n')
    output.write("dev set size: " + str(len(dev_y))+ ' examples\n\n')

    x = tf.placeholder(tf.int32, [None, params['MAX_LENGTH']])
    y_ = tf.placeholder(tf.float32, [None, params['CLASSES']])
    word_embeddings = tf.Variable(tf.convert_to_tensor(key_array, dtype = tf.float32), trainable = params['UPDATE_WORD_VECS'])
    #shape=(len(keys), params['WORD_VECTOR_LENGTH']))
    embedding_layer = tf.nn.embedding_lookup(word_embeddings, x)


    # x encodes data: [batch size, l * word vector length]
    # y_ encodes labels: [batch size, classes]
    # x = tf.placeholder(tf.float32, [params['tensorflow_batch_size'], params['MAX_LENGTH'] * params['WORD_VECTOR_LENGTH']])
    # y_ = tf.placeholder(tf.float32, [params['tensorflow_batch_size'], params['CLASSES']])

    embedding_output = tf.reshape(embedding_layer, [-1, params['MAX_LENGTH'], 1, params['WORD_VECTOR_LENGTH']])

    #init lists for convolutional layer
    slices = []
    weights = []
    biases = []
    #loop over KERNEL_SIZES, each time initializing a slice
    for kernel_size in params['KERNEL_SIZES']:
        slices, weights, biases = define_nn(embedding_output, kernel_size, params, slices, weights, biases)

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

    batches_x, batches_y = get_batches(params, train_x, train_y)

    #run session
    output.write( 'Initializing session...\n\n')
    sess = tf.Session(config=tf.ConfigProto(inter_op_parallelism_threads=2,
                      intra_op_parallelism_threads=3, use_per_session_threads=True))
    sess.run(tf.initialize_all_variables())
    output.write( 'Running session...\n\n')

    best_dev_accuracy = 0
    output.write("initial accuracy %g \n"%accuracy.eval(feed_dict={x: sub_indices(train_x, embed_keys), y_: train_y, dropout: 1.0}, session = \
        sess))
    for i in range(params['EPOCHS']):
        params['epoch'] = i + 1
        for j in range(batches_x.shape[0]/params['BATCH_SIZE']):
            batch_x, batch_y = get_batch(batches_x, batches_y, j, params)
            params['tensorflow_batch_size'] = batch_y.shape[0]
            train_step.run(feed_dict={x: sub_indices(batch_x, embed_keys), y_: batch_y, dropout: params['TRAIN_DROPOUT']}, session = sess)
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
        train_softmax = cross_entropy.eval(feed_dict={x: sub_indices(train_x, embed_keys), y_: train_y, dropout: 1.0}, session = sess)
        output.write("epoch %d, training accuracy %g, training softmax error %g \n"
            %(i, accuracy.eval(feed_dict={x: sub_indices(train_x, embed_keys), y_: train_y, dropout: 1.0}, session = sess), train_softmax))

        dev_softmax = cross_entropy.eval(feed_dict={x: sub_indices(dev_x, embed_keys), y_: dev_y, dropout: 1.0}, session = sess)
        dev_accuracy = accuracy.eval(feed_dict={x: sub_indices(dev_x, embed_keys), y_: dev_y, dropout: 1.0}, session = sess)
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

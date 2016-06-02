#feed_dict rewrites the value of tensors in the graph
#learning rate, decay not mentioned in paper
# implement updating vocab
# word2vec or GloVe pre-trained?
# advantage to typing word2vec as floats or let tf deal w/it?
#fix: get_batch returns only 15 items
#ML txtbk parameter estimation

import tensorflow as tf
import random
import linecache
from text_cnn_methods import *
import numpy as np

#define hyperparameters
def define_globals():
    params = {'WORD_VECTOR_LENGTH' : 300,
        'FILTERS' : 100,
        'KERNEL_SIZES' : [3, 4, 5],
        'CLASSES' : 2,
        'MAX_LENGTH' : 59,

        'L2_NORM_CONSTRAINT' : 3,
        'TRAIN_DROPOUT' : 0.5,

        'TRAINING_STEPS' : 20000,
        'BATCH_SIZE' : 50,

        'TRAIN_FILE_NAME' : 'train-short',
        'DEV_FILE_NAME' : 'dev-short',
        'WORD_VECS_FILE_NAME' : 'output-short.txt',
        'SST' : True,
        'DEV' : False,

        'line_index' : 0,
        #debug
        'key_errors' : []
        }
    return params

params = define_globals()

vocab = find_vocab(params['TRAIN_FILE_NAME'] + '.data', params['SST'])
vocab = find_vocab(params['DEV_FILE_NAME'] + '.data', params['SST'],  vocab=vocab)
keys = initialize_vocab(vocab, params['WORD_VECS_FILE_NAME'])

print list(keys.keys())

train_size = find_lines(params['TRAIN_FILE_NAME'] + '.labels')
dev_size = find_lines(params['DEV_FILE_NAME'] + '.labels')

#keys = {}
#train_size = 500
#dev_size = 500
# x encodes data: [batch size, l * word vector length]
# y_ encodes labels: [batch size, classes]
x = tf.placeholder(tf.float32, [params['BATCH_SIZE'], params['MAX_LENGTH'] * params['WORD_VECTOR_LENGTH']])
y_ = tf.placeholder(tf.float32, [params['BATCH_SIZE'], params['CLASSES']])

#get random batch of examples from test file
def get_batch(lines, params, train_file_list, train_file_labels):
    batch_x = []
    batch_y = []

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
    batch_x = pad(batch_x, params)
    return batch_x, batch_y

print tf.shape(x)[0]
x = tf.reshape(x, [params['BATCH_SIZE'], params['MAX_LENGTH'], 1, params['WORD_VECTOR_LENGTH']])
print tf.shape(x)

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
print slices
h_pool = tf.concat(3, slices)
print h_pool
#apply dropout (p = TRAIN_DROPOUT or TEST_DROPOUT)
dropout = tf.placeholder(tf.float32)
h_pool_drop = tf.nn.dropout(h_pool, dropout)

h_pool_flat = tf.squeeze(h_pool_drop)
#fully connected softmax layer
W_fc = weight_variable([len(params['KERNEL_SIZES']) * params['FILTERS'], params['CLASSES']])
b_fc = bias_variable([params['CLASSES']])
print h_pool_flat
y_conv = tf.nn.softmax(tf.matmul(h_pool_flat, W_fc) + b_fc)

#define error for training steps
#check YK code to find out what he actually uses
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv),
                               reduction_indices=[1]))
train_step = tf.train.AdadeltaOptimizer().minimize(cross_entropy)

#define accuracy for evaluation
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

#create training and eval sets
dev_x,dev_y = get_all(params['DEV_FILE_NAME'], dev_size, keys, params)
print len(params['key_errors'])
print params['key_errors']
train_x,train_y = get_all(params['TRAIN_FILE_NAME'], train_size, keys, params)
print len(params['key_errors'])
print params['key_errors']
shuffle_x,shuffle_y = shuffle(train_x,train_y)
dev_x = np.asarray(dev_x)
dev_y = np.asarray(dev_y)
train_x = np.asarray(train_x)
train_y = np.asarray(train_y)
shuffle_x = np.asarray(shuffle_x)
shuffle_y = np.asarray(shuffle_y)

#run session
print "Initializing session..."
print ''
sess = tf.Session(config=tf.ConfigProto(inter_op_parallelism_threads=1,
                  intra_op_parallelism_threads=10, use_per_session_threads=True))
sess.run(tf.initialize_all_variables())
print 'Running session...'
print ''
for i in range(params['TRAINING_STEPS']):
    batch_x, batch_y = get_batch(train_size, params, shuffle_x, shuffle_y)
    if i%100 == 0:
        train_accuracy = accuracy.eval(feed_dict={
            x: batch_x, y_: batch_y, dropout: 1.0}, session = sess)
        print("step %d, training accuracy %g"%(i, train_accuracy))
    #prints accuracy for dev set every 1000 examples, DEV is a hyperparameter boolean
    if DEV and i%1000 == 0:
        print("dev set accuracy %g"%accuracy.eval(feed_dict={
            x: params['DEV_FILE_NAME'] + '.data',
            y_: params['DEV_FILE_NAME'] + '.labels',
            dropout: 1.0},
            session = sess))
    train_step.run(feed_dict={x: batch_x, y_: batch_y, dropout: params['TRAIN_DROPOUT']})

    #normalize weights
    for W in weights:
        W = l2_normalize(W, params[L2_NORM_CONSTRAINT])
    W_fc = l2_normalize(W_fc, params[L2_NORM_CONSTRAINT])
    #normalize biases
    for b in biases:
        b = l2_normalize(b, params[L2_NORM_CONSTRAINT])
    b_fc = l2_normalize(b_fc, params[L2_NORM_CONSTRAINT])

#print test accuracy of results
print("test accuracy %g"%accuracy.eval(feed_dict={x: train_x, y_: train_y, dropout: 1.0}, session = sess))
#print dev accuracy of results
if DEV:
    print("dev set accuracy %g"%accuracy.eval(feed_dict={x: dev_x, y_: dev_y, dropout: 1.0}, session = sess))

if __name__ == "__main__": main()

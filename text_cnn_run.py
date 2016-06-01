#feed_dict rewrites the value of tensors in the graph
#learning rate, decay not mentioned in paper
# implement updating vocab
# choose same line twice ok?
# flags?
#params dictionary
# replicate random of YKim

import tensorflow as tf
import random
import linecache
from text_cnn_methods import *

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

        'TRAIN_FILE_NAME' : 'train',
        'DEV_FILE_NAME' : 'dev',
        'WORD_VECS_FILE_NAME' : 'output-short.txt',
        'SST' : True,
        'DEV' : False }
    return params
"""
vocab,train_size = find_vocab(TRAIN_FILE_NAME + '.data', SST)
vocab,dev_size = find_vocab(DEV_FILE_NAME + '.data', SST,  vocab=vocab)
keys = initialize_vocab(vocab, WORD_VECS_FILE_NAME)
"""
params = define_globals()
print params
keys = {}
train_size = 500
dev_size = 500
# x encodes data: [batch size, l * word vector length]
# y_ encodes labels: [batch size, classes]
x = tf.placeholder(tf.float32, [params['BATCH_SIZE'], params['MAX_LENGTH'] * params['WORD_VECTOR_LENGTH']])
y_ = tf.placeholder(tf.float32, [params['BATCH_SIZE'], params['CLASSES']])
line_index = 0
train_file_list,train_file_labels = shuffle_file(params, train_size, keys)
#get random batch of examples from test file
def get_batch(lines, params, train_file_list, train_file_labels):
    batch_x = []
    batch_y = []
    if line_index + params['BATCH_SIZE'] <= lines:
        for line in range(line_index, line_index + params['BATCH_SIZE']):
            batch_x.append(train_file_list[line])
            print train_file_labels[line]
            print batch_y
            batch_y.append(train_file_labels[line])
        params[line_index] += params['BATCH_SIZE']
    else:
        counter = 0
        while counter + line_index <= lines:
            batch_x.append(train_file_list[line])
            batch_y.append(train_file_labels[line])
        params[line_index] = params['BATCH_SIZE'] - counter
        while counter < params['BATCH_SIZE']:
            batch_x.append(train_file_list[line])
            batch_y.append(train_file_labels[line])
        #get random line index in file
        #line_index = random.randrange(lines)
        #batch_x.append(line_to_vec(linecache.getline(file_name + '.data',
        #               line_index), keys, WORD_VECTOR_LENGTH))
        #get label and turn into one-hot vector
        #batch_y.append(one_hot(int(linecache.getline(file_name + '.labels',
        #               line_index), CLASSES)))
    batch_x = pad(batch_x, params)
    return batch_x, batch_y

def get_all(file_name, lines, params):
    all_x = []
    all_y = []
    text_file = open(file_name, 'r')
    labels = open(file_name + '.labels', 'r')
    for line in lines:
        all_x.insert(len(all_x), line_to_vec(text_file.readline().replace(':', ''), keys,
                     params['WORD_VECTOR_LENGTH']))
        all_y.insert(len(all_y), one_hot(int(labels.readline().rstrip()), params['CLASSES']))
    all_x = pad(batch_x, params)
    return all_x, all_y

print tf.shape(x)[0]
x = tf.reshape(x, [params['BATCH_SIZE'], params['MAX_LENGTH'], 1, params['WORD_VECTOR_LENGTH']])
print tf.shape(x)
#loop over KERNEL_SIZES, each time initializing a slice
try: slices
except NameError: slices = []
try: weights
except NameError: weights = []
try: biases
except NameError: biases = []
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
dev_x,dev_y = get_all(params['DEV_FILE_NAME'], dev_size, )
#run session
sess = tf.Session(config=tf.ConfigProto(inter_op_parallelism_threads=1,
                  intra_op_parallelism_threads=10, use_per_session_threads=True))
sess.run(tf.initialize_all_variables())
for i in range(params['TRAINING_STEPS']):
    batch_x, batch_y = get_batch(train_size, params, train_file_list, train_file_labels)
    if i%100 == 0:
        train_accuracy = accuracy.eval(feed_dict={
            x: batch_x, y_: batch_y, dropout: 1.0})
        print("step %d, training accuracy %g"%(i, train_accuracy))
    #prints accuracy for dev set every 1000 examples, DEV is a hyperparameter boolean
    if DEV and i%1000 == 0:
        print("dev set accuracy %g"%accuracy.eval(feed_dict={
            x: params['DEV_FILE_NAME'] + '.data', y_: params['DEV_FILE_NAME'] + '.labels', dropout: 1.0}))
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
all_x, all_y = get_all(TRAIN_FILE_NAME, train_size)
print("test accuracy %g"%accuracy.eval(feed_dict={x: all_x, y_: all_y, dropout: 1.0}))
#print dev accuracy of results
if DEV:
    all_x, all_y = get_all(DEV_FILE_NAME, dev_size)
    print("dev set accuracy %g"%accuracy.eval(feed_dict={x: all_x, y_: all_y, dropout: 1.0}))

if __name__ == "__main__": main()

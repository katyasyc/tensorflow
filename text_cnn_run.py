# problems: backend
#feed_dict rewrites the value of tensors in the graph
#learning rate, decay not mentioned in paper
#clean up get_example
#fix the calls to files at the bottom: mnist convnet was looking at methods, not filenames!!
# implement updating vocab

import tensorflow as tf
from random import *
from text_cnn_methods import *

#define hyperparameters
WORD_VECTOR_LENGTH = 300
FILTERS = 100
KERNEL_SIZES = [3, 4, 5]
CLASSES = 2
MAX_LENGTH = 59

L2_NORM_CONSTRAINT = 3
TRAIN_DROPOUT = 0.5

TRAINING_STEPS = 20000
BATCH_SIZE = 50

TRAIN_FILE_NAME = 'train'
DEV_FILE_NAME = 'dev'
WORD_VECS_FILE_NAME = 'output-short.txt'
DEV = False

vocab,train_size = find_vocab(TRAIN_FILE_NAME + '.data')
vocab,dev_size = find_vocab(DEV_FILE_NAME + '.data',  vocab=vocab)
keys = initialize_vocab(vocab, WORD_VECS_FILE_NAME)

length = MAX_LENGTH + max(KERNEL_SIZES) - max(KERNEL_SIZES) % 2
keys = {}
# x encodes data: [batch size, l * word vector length]
# y_ encodes labels: [batch size, classes]
x = tf.placeholder(tf.float32, [BATCH_SIZE, length * WORD_VECTOR_LENGTH])
y_ = tf.placeholder(tf.float32, [BATCH_SIZE, CLASSES])

#get random batch of examples from test file
def get_batch(file_name, lines):
    batch_x = []
    batch_y = []
    length = 0
    for i in range(BATCH_SIZE):
        #get random line index in file
        line_index = random.randrange(lines)
        batch_x.append(line_to_vec(linecache.getline(file_name + '.data',
                       line_index), keys, WORD_VECTOR_LENGTH, length))
        #length = max(max_length, len(batch_x[i])/300)
        #get label and turn into one-hot vector: store in python list for now
        batch_y.append(one_hot(int(linecache.getline(file_name + '.labels',
                       line_index), CLASSES)))
    #batch_x = pad(batch_x, length, WORD_VECTOR_LENGTH)
    #convert to tensors: 2d [batch length, sample length] (for x) or [~ CLASSES] (for y)]
    return batch_x, batch_y

def get_all(file_name, lines):
    all_x = []
    all_y = []
    text_file = open(file_name, 'r')
    for line in lines:
        all_x.append(line_to_vec(text_file.readline().replace(':', ''), keys,
                     WORD_VECTOR_LENGTH, max(KERNEL_SIZES)/2))
        all_y.append(tf.one_hot(int(linecache.getline(file_name + '.labels',
                     line_index), CLASSES, 1, 0)))
    all_x = pad(batch_x, length, WORD_VECTOR_LENGTH)
    return all_x, all_y

print tf.shape(x)[0]
x = tf.reshape(x, [BATCH_SIZE, length, 1 ,WORD_VECTOR_LENGTH])
print tf.shape(x)
#loop over KERNEL_SIZES, each time initializing a slice
try: slices
except NameError: slices = []
try: weights
except NameError: weights = []
try: biases
except NameError: biases = []
for kernel_size in KERNEL_SIZES:
    slices, weights, biases = define_nn(x, kernel_size, length, FILTERS, WORD_VECTOR_LENGTH, slices, weights, biases)
print slices
h_pool = tf.concat(3, slices)
print h_pool
#apply dropout (p = TRAIN_DROPOUT or TEST_DROPOUT)
dropout = tf.placeholder(tf.float32)
h_pool_drop = tf.nn.dropout(h_pool, dropout)

h_pool_flat = tf.squeeze(h_pool_drop)
#fully connected softmax layer
W_fc = weight_variable([len(KERNEL_SIZES) * FILTERS, CLASSES])
b_fc = bias_variable([CLASSES])
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

#run session
sess = tf.Session(config=tf.ConfigProto(inter_op_parallelism_threads=1,
                  intra_op_parallelism_threads=10, use_per_session_threads=True))
sess.run(tf.initialize_all_variables())
for i in range(TRAINING_STEPS):
    batch_x, batch_y = get_batch(TRAIN_FILE_NAME, train_size)
    if i%100 == 0:
        train_accuracy = accuracy.eval(feed_dict={
            x: batch_x, y_: batch_y, dropout: 1.0})
        print("step %d, training accuracy %g"%(i, train_accuracy))
    #prints accuracy for dev set every 1000 examples, DEV is a hyperparameter boolean
    if DEV and i%1000 == 0:
        print("dev set accuracy %g"%accuracy.eval(feed_dict={
            x: DEV_FILE_NAME + '.data', y_: DEV_FILE_NAME + '.labels', dropout: 1.0}))
    train_step.run(feed_dict={x: batch_x, y_: batch_y, dropout: TRAIN_DROPOUT})

    #normalize weights
    for W in weights:
        W = l2_normalize(W, L2_NORM_CONSTRAINT)
    W_fc = l2_normalize(W_fc, L2_NORM_CONSTRAINT)
    #normalize biases
    for b in biases:
        b = l2_normalize(b, L2_NORM_CONSTRAINT)
    b_fc = l2_normalize(b_fc, L2_NORM_CONSTRAINT)

#print test accuracy of results
all_x, all_y = get_all(TRAIN_FILE_NAME, train_size)
print("test accuracy %g"%accuracy.eval(feed_dict={x: all_x, y_: all_y, dropout: 1.0}))
#print dev accuracy of results
if DEV:
    all_x, all_y = get_all(DEV_FILE_NAME, dev_size)
    print("dev set accuracy %g"%accuracy.eval(feed_dict={x: all_x, y_: all_y, dropout: 1.0}))

if __name__ == "__main__": main()

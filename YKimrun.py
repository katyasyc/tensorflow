# problems: backend
#feed_dict rewrites the value of tensors in the graph
#learning rate, decay not mentioned in paper
#clean up get_example

import tensorflow as tf
from libs.utils import *
from random import *
import matplotlib.pyplot as plt
import file2vec.py

#define hyperparameters
WORD_VECTOR_LENGTH = tf.constant(300)
FILTERS = tf.constant(100)
KERNEL_SIZES = [3, 4, 5]
CLASSES = tf.constant(2)

L2_NORM_CONSTRAINT = tf.constant(3)
TRAIN_DROPOUT = tf.constant(0.5)

TRAINING_STEPS = tf.constant(tf.int32, [20000])
BATCH_SIZE = tf.constant(tf.int32, [50])

TRAIN_FILE_NAME = train
DEV_FILE_NAME = dev
DEV = FALSE

dev_lines = find_lines(DEV_FILE_NAME.labels)
train_lines = find_lines(TRAIN_FILE_NAME.labels)

# x encodes data: [batch size, l * word vector length]
# y encodes labels: [batch size, classes]
x = tf.placeholder(tf.float32, [BATCH_SIZE, None])
y = tf.placeholder(tf.int32, [BATCH_SIZE, CLASSES])

def find_lines(file_name):
    for i, l in enumerate(file_name):
        pass
    return i

#get random batch of examples from test file
def get_batch(file_name, lines):
    batch_x = []
    batch_y = []

    length = 0
    for i in range(BATCH_SIZE)
        #get random line index in file
        line_index = random.randrange(lines)
        batch_x.append(file2vec.main(linecache.getline(file_name.data,
            line_index), WORD_VECTOR_LENGTH, max(KERNEL_SIZES)/2)
        length = max(max_length, batch_x[i])
        #get label and turn into one-hot vector—store in python list for now
        batch_y.append(tf.one_hot(int(linecache.getline(file_name.labels,
            line_index)], CLASSES, 1, 0)))
    #check y = batch_y
    for sample in batch_x:
        left = (length - tf.size(sample)) / 2
        right = left
        if (length - tf.size(sample)) % 2 != 0:
            right += 1
        sample = sample.insert(0, [0] * WORD_VECTOR_LENGTH * left)
        sample = sample.extend([0] * WORD_VECTOR_LENGTH * right)
    #convert to tensors: 2d [batch length, sample length (for x) or CLASSES (for y)]
    x = batch_x
    y = batch_y
    return x, y

#l2_loss = l2 loss (tf fn returns half of l2 loss w/o sqrt)
#where Wi is each item in W, W = Wi/sqrt[sum([(Wi*constraint)/l2_loss]^2)]
def l2_normalize(W):
    l2_loss = sqrt(2*tf.nn.L2_loss(W))
    if  l2_loss > L2_NORM_CONSTRAINT:
        W = tf.scalar_mul(1/sqrt(tf.reduce_sum(tf.square(tf.scalar_mul(L2_NORM_CONSTRAINT/l2_loss, W), 2))), W)
    return W

#I don't think appending will work here— i don't understand that part of the code
def define_nn(kernel_size)
    #define weights and biases—make sure we can specify to normalize later
    W = tf.truncated_normal([kernel_size, None, 1, FILTERS], stddev=0.1)
    b = bias_variable(tf.constant(0.1, [FILTERS]=shape))
    #convolve—each neuron iterates by 1 filter, 1 word
    conv = tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding="SAME")
    #apply bias and relu
    relu = tf.nn.relu(tf.nn.bias_add(conv, b))
    #max pool; each neuron sees 1 filter and returns max over l
    pooled = tf.nn.max_pool(relu, ksize=[1, None, 1, 1],
        strides=[1, None, 1, 1], padding='SAME')
    slices.append(pooled)
    weights.append(W)
    biases.append(b)

#fill by looping over KERNEL_SIZES, each time initializing a slice
slices = []
weights = []
biases = []

for kernel_size in enumerate(KERNEL_SIZES):
    define_nn(kernel_size)

#combine all slices in a single vector
h_pool = tf.concat(2, slices)
h_pool_flat = tf.reshape(h_pool, [-1, len(KERNEL_SIZES) * FILTERS])

#apply dropout (p = TRAIN_DROPOUT or TEST_DROPOUT)
dropout = tf.placeholder(tf.float32)
h_pool_drop = tf.nn.dropout(h_pool, keep_prob)

#fully connected softmax layer
W_fc = tf.truncated_normal([len(KERNEL_SIZES) * FILTERS, CLASSES], stddev=0.1)
b_fc = bias_variable(tf.constant(0.1, [CLASSES]=shape))
y_conv = tf.nn.softmax(tf.matmul(h_pool_drop, W_fc) + b_fc)

#define error for training steps
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
for i in TRAINING_STEPS:
  batch_x, batch_y = get_batch(TRAIN_FILE_NAME, train_lines)
  if i%100 == 0:
    train_accuracy = accuracy.eval(feed_dict={
        x:batch[0], y_: batch[1], dropout: 1.0})
    print("step %d, training accuracy %g"%(i, train_accuracy))
  #prints accuracy for dev set every 1000 examples, DEV is a hyperparameter boolean
  if DEV and i%1000 = 0:
      print("dev set accuracy %g"%accuracy.eval(feed_dict={
          x: DEV_FILE_NAME.data, y_: DEV_FILE_NAME.labels, dropout: 1.0}))
  train_step.run(feed_dict={x: batch_x, y_: batch_y, dropout: TRAIN_DROPOUT})

  #normalize weights
  for W in weights:
      W = l2_normalize(W)
  W_fc = l2_normalize(W_fc)
  #normalize biases
  for b in biases:
      b = l2_normalize(b)
  b_fc = l2_normalize(b_fc)

#print test accuracy of results
print("test accuracy %g"%accuracy.eval(feed_dict={
    x: TRAIN_FILE_NAME.data, y_: TRAIN_FILE_NAME.labels, dropout: 1.0}))
#print dev accuracy of results
if DEV:
    print("dev set accuracy %g"%accuracy.eval(feed_dict={
        x: DEV_FILE_NAME.data, y_: DEV_FILE_NAME.labels, dropout: 1.0}))

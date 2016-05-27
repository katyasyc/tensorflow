# problems: backend
# vectors.data needs to be test or something—separate dev and test sets

#imports
import tensorflow as tf
from libs.utils import *
import matplotlib.pyplot as plt
from YKim_open_examples.py import get_examples
#import our data

#define hyperparameters
WORD_VECTOR_LENGTH = tf.constant(300)
FILTERS = tf.constant(100)
KERNEL_SIZES = tf.placeholder(tf.int32, [3, 4, 5])
CLASSES = tf.constant(2)

L2_NORM_CONSTRAINT = tf.constant(3)
TRAIN_DROPOUT = tf.constant(0.5)
TEST_DROPOUT = tf.constant(1.0)

TRAINING_STEPS = tf.constant(tf.int32, [20000])
BATCH_SIZE = tf.constant(tf.int32, [50])

TEST_FILE_NAME = vectors.data
DEV_FILE_NAME

#create tensors for values and labels
x = tf.placeholder(tf.float32, [None, WORD_VECTOR_LENGTH * l])
y = tf.placeholder(tf.int32, [None, 2])

#given a BATCH_SIZE, get examples from test file
def get_batch(BATCH_SIZE, file_name)
    #create an empty string to store our batch of vectors
    batch = ''
    #import test file
    examples = open(file_name, 'r')
    #get file size
    examples.seek(0, os.SEEK_END)
    filesize = examples.tell()
    #go to a random index in file
    for i in xrange(BATCH_SIZE):
        examples.seek(randrange(0, filesize))
        examples.readline()
        batch.append(examples.readline())
    return batch

#not sure if this will work, see http://stackoverflow.com/questions/33944683/tensorflow-map-operation-for-tensor
def l2_normalize(W):
    l2_loss = sqrt(tf.scalar_mul(2,tf.nn.L2_loss(W))
    if  l2_loss > L2_NORM_CONSTRAINT:
        W = sqrt(tf.scalar_mul(L2_NORM_CONSTRAINT/l2_loss, W)^2)

def define_nn(kernel_size)
    #define weights and biases—make sure we can specify to normalize later
    W = tf.truncated_normal([kernel_size, l, 1, FILTERS], stddev=0.1)
    b = bias_variable(tf.constant(0.1, [FILTERS]=shape))
    #convolve—each neuron iterates by 1 filter, 1 word
    conv = tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding="SAME")
    #apply bias and relu
    relu = tf.nn.relu(tf.nn.bias_add(conv, b))
    #max pool; each neuron sees 1 filter and returns max over l
    pooled = tf.nn.max_pool(relu, ksize=[1, l, 1, 1],
        strides=[1, l, 1, 1], padding='SAME')
    slices.append(pooled)
    weights.append(W)
    biases.append(b)

#fill slices[] by looping over KERNEL_SIZES, each time running the network
#keep weights and biases to modify later
slices = []
weights = []
biases = []

for kernel_size in enumerate(KERNEL_SIZES):
    define_nn(kernel_size, l, FILTERS)

#combine all slices in a vector
h_pool = tf.concat(len(KERNEL_SIZES), slices)
h_pool_flat = tf.reshape(h_pool, [-1, len(KERNEL_SIZES) * FILTERS])

#apply dropout (p = TRAIN_DROPOUT or TEST_DROPOUT)
#tf.nn.dropout op automatically scales neuron outputs in addition to masking them
dropout = tf.placeholder(tf.float32)
h_pool_drop = tf.nn.dropout(h_pool, keep_prob)

#send max pooled, dropped out vector to a fully connected softmax layer with CLASSES classes
W_fc = tf.truncated_normal([len(KERNEL_SIZES) * FILTERS, CLASSES], stddev=0.1)
b_fc = bias_variable(tf.constant(0.1, [CLASSES]=shape))

y_conv=tf.nn.softmax(tf.matmul(h_pool_drop, W_fc) + b_fc)

#define error for training steps
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv), reduction_indices=[1]))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
#define accuracy for evaluation
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

#run session
sess = tf.Session()
#these look like they require imports
sess.run(tf.initialize_all_variables())
for i in TRAINING_STEPS:
    #sst2vec = name of file we are working from
  batch = sst2vec.train.get_batch(BATCH_SIZE,TEST_FILE_NAME)
  if i%100 == 0:
    train_accuracy = accuracy.eval(feed_dict={
        x:batch[0], y_: batch[1], dropout: 1.0})
    print("step %d, training accuracy %g"%(i, train_accuracy))
  #feed_dict is an optional argument that overrides the value of tensors in the graph
  train_step.run(feed_dict={x: batch[0], y_: batch[1], dropout: TRAIN_DROPOUT})

  #normalize weights
  for W in weights:
      W = l2_normalize(W)
  W_fc = l2_normalize(W_fc)
  #normalize biases
  for b in biases:
      b = l2_normalize(b)
  b_fc = l2_normalize(b_fc)

#print accuracy of results
print("test accuracy %g"%accuracy.eval(feed_dict={
    x: sst2vec.test.strings, y_: sst2vec.test.labels, dropout: TEST_DROPOUT}))

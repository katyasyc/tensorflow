# problems: l is a Variable
# how to incorporate l2 normalization
# is std in backend file a form of regularization?
# int64 labels to int100 labels ??
# what happened to max pooling layer in backend file??

#imports
import tensorflow as tf
import tensorflow.examples.tutorials.mnist.input_data as input_data
from libs.utils import *
import matplotlib.pyplot as plt

#hyperparameters
WORD_VECTOR_LENGTH = 300
FILTERS = 100
#KERNEL_SIZES = [3,4,5]
CLASSES = 2

TRAINING_STEPS = 20000
BATCH_SIZE = 50
L2_NORM_CONSTRAINT = 3

#initializes weights
def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

#initializes biases
def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

#stride of one and zero padding so that there are the same # of neurons as words in sample
def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1], padding='SAME')

#max pooling layer: each neuron looks at 1 conv slice and 1 filter,
#returns maximum over entire sample length/word count (l)
# ksize and strides over [1, number of conv slices, l, filters]
def max_pool(x):
  return tf.nn.max_pool(x, ksize=[1, 1, l, 1],
                        strides=[1, 1, l, 1], padding='SAME')

#not sure if this will work, see http://stackoverflow.com/questions/33944683/tensorflow-map-operation-for-tensor
def l2_normalize(w):
    l2_loss = sqrt(tf.scalar_mul(2,tf.nn.L2_loss(w))
    if  l2_loss > L2_NORM_CONSTRAINT:
        w = sqrt(tf.scalar_mul(L2_NORM_CONSTRAINT/l2_loss,w)^2)

#weights for first convolutional slice: [kernel size, 1, word vector length, filters]
W_conv3 = weight_variable([3, 1, WORD_VECTOR_LENGTH, FILTERS])
#bias vector has channel for each output channel
b_conv3 = bias_variable([FILTERS])

#reshape x to a 3d tensor, 2nd dimension corresponds to length = word count
#the final dimension corresponds to the # input channels (length of word2vec).
x_samples = tf.reshape(x, [-1, l, 1, WORD_VECTOR_LENGTH])
#convolve input vectors with the weight tensor,
#add the bias, apply the ReLU function
h_conv3 = tf.nn.relu(conv2d(x_samples, W_conv3) + b_conv3)

#weights for 2nd convolutional slice: kernel size = 4
#make sure that it has same # neurons as # words!!!
W_conv4 = weight_variable([4, 1, WORD_VECTOR_LENGTH, FILTERS])
b_conv4 = bias_variable([FILTERS])
h_conv4 = tf.nn.relu(conv2d(x_samples, W_conv4) + b_conv4)

#weights for 3rd convolutional slice: kernel size = 5
W_conv5 = weight_variable([5, 1, WORD_VECTOR_LENGTH, FILTERS])
b_conv5 = bias_variable([FILTERS])
h_conv5 = tf.nn.relu(conv2d(x_samples, W_conv5) + b_conv5)

#reshape outputs of conv slices so we can pool them all together
#would it make sense to concatenate them, getting [3,l,FILTERS]?
h_conv3 = tf.reshape(h_conv3, [1, l, FILTERS])
h_conv4 = tf.reshape(h_conv4, [1, l, FILTERS])
h_conv5 = tf.reshape(h_conv5, [1, l, FILTERS])

#max pool over convolutional slices
h_pool = max_pool(h_conv3, h_conv4, h_conv5)
#returns a 2d tensor [3, 100] corresponding to the 3 conv slices and 100 feature maps

#reshape output of pooling layer into a vector
h_pool_flat = tf.reshape(h_pool, [-1, 3*FILTERS])

#apply dropout before the readout layer.
#placeholder for probability that a neuron's output is kept during dropout.
#therefore can turn dropout on during training and turn it off during testing.
#tf.nn.dropout op automatically scales neuron outputs in addition to masking them, so dropout just works without any additional scaling.
keep_prob = tf.placeholder(tf.float32)
h_pool_drop = tf.nn.dropout(h_pool, keep_prob)

#send max pooled, dropped out vector to a fully connected softmax layer with binary options
W_fc = weight_variable([FILTERS*3, CLASSES])
b_fc = bias_variable([CLASSES])

y_conv=tf.nn.softmax(tf.matmul(h_pool_drop, W_fc) + b_fc)

#train
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv), reduction_indices=[1]))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

sess = tf.Session()
sess.run(tf.initialize_all_variables())
for i in range(TRAINING_STEPS):
    #sst2vec = name of file we are working from
  batch = sst2vec.train.next_batch(BATCH_SIZE)
  if i%100 == 0:
    train_accuracy = accuracy.eval(feed_dict={
        x:batch[0], y_: batch[1], keep_prob: 1.0})
    print("step %d, training accuracy %g"%(i, train_accuracy))
  train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

  #normalize weights
  l2_normalize(W_conv3)
  l2_normalize(W_conv4)
  l2_normalize(W_conv5)
  l2_normalize(W_fc)
  #normalize biases
  l2_normalize(b_conv3)
  l2_normalize(b_conv4)
  l2_normalize(b_conv5)
  l2_normalize(b_fc)


#print accuracy of results
print("test accuracy %g"%accuracy.eval(feed_dict={
    x: sst2vec.test.strings, y_: sst2vec.test.labels, keep_prob: 1.0}))

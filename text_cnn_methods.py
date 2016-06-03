import tensorflow as tf
import sys, re
import random
import numpy as np
import math

#initializes weights, random with stddev of .1
def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

#initializes biases, all at .1
def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

#defines the first two layers of our neural network
def define_nn(x, kernel_size, params, slices, weights, biases):
    #define weights and biases, make sure we can specify to normalize later
    #correct line: getting error
    #2nd dimension should be "None"
    #fix kernel_size
    W = weight_variable([kernel_size, 1, params['WORD_VECTOR_LENGTH'], params['FILTERS']])
    b = bias_variable([params['FILTERS']])
    #convolve: each neuron iterates by 1 filter, 1 word
    conv = tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding="SAME")
    #apply bias and relu
    relu = tf.nn.relu(tf.nn.bias_add(conv, b))
    #max pool; each neuron sees 1 filter and returns max over l
    pooled = tf.nn.max_pool(relu, ksize=[1, params['MAX_LENGTH'], 1, 1],
        strides=[1, params['MAX_LENGTH'], 1, 1], padding='SAME')
    slices.insert(len(slices), pooled)
    weights.insert(len(weights), W)
    biases.insert(len(biases), b)
    return slices, weights, biases

def one_hot(category, CLASSES):
    one_hot = [0] * CLASSES
    one_hot[category] = 1
    return one_hot

#super buggy
def pad(sample, params):
    left = (params['MAX_LENGTH']  - len(sample)) / 2
    right = left
    if (params['MAX_LENGTH'] - len(sample)) % 2 != 0:
        right += 1
    for i in range(left):
        sample.insert(0, [0] * params['WORD_VECTOR_LENGTH'])
    for i in range(right):
        sample.append([0] * params['WORD_VECTOR_LENGTH'])
    return sample

#l2_loss = l2 loss (tf fn returns half of l2 loss w/o sqrt)
#where Wi is each item in W, W = Wi/sqrt[sum([(Wi*constraint)/l2_loss]^2)]
def l2_normalize(W, L2_NORM_CONSTRAINT, sess):
    print 'w', W
    loss =tf.identity(tf.nn.l2_loss(W))
    print 'w2', W
    print 'loss', loss
    loss2 = tf.cast(loss, dtype = tf.float32)
    print 'loss2', loss2
    l2_loss = math.sqrt(2*(sess.run(loss2)))
    """
    if  l2_loss > L2_NORM_CONSTRAINT:

        """
    # if  l2_loss > L2_NORM_CONSTRAINT:
    #     W = tf.scalar_mul(tf.rsqrt(tf.reduce_sum(tf.square(
    #         tf.scalar_mul(tf.convert_to_tensor(L2_NORM_CONSTRAINT/l2_loss, as_ref = True),
    #         tf.convert_to_tensor(W, as_ref = True)), tf.convert_to_tensor(2)))), W)
    if  l2_loss > L2_NORM_CONSTRAINT:
        #T = tf.Variable(tf.convert_to_tensor(, as_ref = True))
        #print T
        """
        T = tf.Variable(L2_NORM_CONSTRAINT/l2_loss)
        print T
        F = tf.constant(2, dtype = tf.float32_ref)
        print F
        """
        #working:
        dummy = tf.constant(0.0)
        T = tf.constant(L2_NORM_CONSTRAINT/l2_loss)
        F = tf.add(dummy, T)
        print F
        W = tf.convert_to_tensor(W, as_ref = True)
        G = tf.cast(F, dtype = tf.float32_ref)
        I = tf.scalar_mul(G, W)
        I1 = tf.cast(I, dtype = tf.float32_ref)
        #print tf.convert_to_tensor(L2_NORM_CONSTRAINT/l2_loss)
        #I1 = tf.scalar_mul(T, W)
        I2 = tf.cast(tf.rsqrt(tf.reduce_sum(tf.square(
            I1))), dtype = tf.float32_ref)
        W = tf.scalar_mul(I2, W)

        #scalar_tensor = tf.cast(tf.constant(L2_NORM_CONSTRAINT/l2_loss), dtype = float32_ref)
        #new_weights =
        #scalar_tensor_2 = tf.cast(tf.rsqrt(tf.reduce_sum(tf.square(I1))), dtype = float32_ref)

    return W

#get all examples from a file and return np arrays w/input and output
def get_all(file_name, lines, d, params):
    input_file = open(file_name + '.data', 'r')
    output_file = open(file_name + '.labels', 'r')
    input_list = []
    output_list = []
    #change this code: we vectorize only as needed
    for line in range(lines):
        try: input_list.append(line_to_vec(input_file.readline(), d, params))
        #debug code: fixed
        except KeyError:
            params['key_errors'].append(clean_str(input_file.readline(), params['SST']))
        #end debug code
        output_list.append(one_hot(int(output_file.readline().rstrip()), params['CLASSES']))
        input_list[line] = pad(input_list[line], params)
    return np.expand_dims(np.asarray(input_list), 2), np.asarray(output_list)

#shuffle two numpy arrays in unison
def shuffle_in_unison(a, b):
    rng_state = np.random.get_state()
    np.random.shuffle(a)
    np.random.set_state(rng_state)
    np.random.shuffle(b)
    return a, b

#defunct
def shuffle(input_list, output_list):
    z = zip(input_list, output_list)
    random.shuffle(z)
    print type(z)
    #actually tuples!
    input_list, output_list = zip(*z)
    input_list = list(input_list)
    output_list = list(output_list)
    return input_list, output_list

#takes a line of text, returns an array of strings where ecah string is a word
def tokenize(line):
   list_of_words = []
   word = ''
   for char in line:
      if char == ' ':
         list_of_words.append(word)
         word = ''
      else:
         word += char
   list_of_words.append(word.strip())
   return list_of_words

#imported from
def clean_str(string, TREC=False, SST=False):
    if SST == True:
        """
        Tokenization/string cleaning for the SST dataset
        """
        string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
        string = re.sub(r"\s{2,}", " ", string)
        return string.strip().lower()
    """
    Tokenization/string cleaning for all datasets except for SST.
    Every dataset is lower cased except for TREC
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip() if TREC else string.strip().lower()

#takes a line of text, key with vocab indexed to vectors
#returns word vectors concatenated into a list
def line_to_vec(sample, d, params):
    list_of_words = tokenize(clean_str(sample, SST = params['SST']))
    word_vectors = []
    for word in list_of_words:
        word_vectors.append(d[word])
    return word_vectors

def find_lines(file_name):
    text_file = open(file_name, 'r')
    temp_string = text_file.read()
    return temp_string.count('\n')

#create a vocabulary list from a file
def find_vocab(file_name, SST, vocab=None, master_key=None):
    if vocab is None:
        vocab = []
    if master_key is None:
        master_key = {}
    text_file = open(file_name, 'r')
    list_of_words = tokenize(clean_str(text_file.read(), SST=SST))
    for word in list_of_words:
        if word not in master_key and word not in vocab:
            vocab.append(word)
    return vocab

#initialize list of vocabulary with word2vec or zeroes
def initialize_vocab(vocab, word_vectors, master_key=None):
    print "vocab size: " + str(len(vocab))
    if master_key is None:
        master_key = {}
    word2vec = open(word_vectors, 'r')
    word2vec.readline()
    for i in range(3000000):   #number of words in word2vec
        line = tokenize(word2vec.readline().strip())
        #turn into floats
        if line[0] in vocab:
            vector = []
            for j in range(1, len(line)):
                vector.append(float(line[j]))
            master_key[vocab[vocab.index(line[0])]] = vector
            vocab.remove(line[0])
    for word in vocab:
        master_key[word] = [0] * 300
    return master_key

if __name__ == "__main__": main()

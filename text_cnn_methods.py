import tensorflow as tf
import sys, re
import random

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
    print kernel_size
    W = weight_variable([kernel_size, 1, params['WORD_VECTOR_LENGTH'], params['FILTERS']])
    b = bias_variable([params['FILTERS']])
    #convolve: each neuron iterates by 1 filter, 1 word
    print tf.shape(x)
    print tf.size(x)
    conv = tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding="SAME")
    #apply bias and relu
    relu = tf.nn.relu(tf.nn.bias_add(conv, b))
    print relu
    #max pool; each neuron sees 1 filter and returns max over l
    pooled = tf.nn.max_pool(relu, ksize=[1, length, 1, 1],
        strides=[1, params['MAX_LENGTH'], 1, 1], padding='SAME')
    slices.insert(len(slices), pooled)
    weights.insert(len(weights), W)
    biases.insert(len(biases), b)
    return slices, weights, biases

def one_hot(category, CLASSES):
    one_hot = [0] * CLASSES
    one_hot[category] = 1
    return one_hot

#sample should be a list, but it's being a string :(
def pad(batch_x, params):
    print batch_x
    print type(batch_x)
    print len(batch_x)
    print batch_x[0]
    print type(batch_x[0])
    for sample in batch_x:
        left = (params['MAX_LENGTH'] - len(sample)) / 2
        right = left
        if (params['MAX_LENGTH'] - len(sample)) % 2 != 0:
            right += 1
        sample = sample.insert(0, [0] * params['WORD_VECTOR_LENGTH'] * left)
        sample = sample.extend([0] * params['WORD_VECTOR_LENGTH'] * right)
    return batch_x

#l2_loss = l2 loss (tf fn returns half of l2 loss w/o sqrt)
#where Wi is each item in W, W = Wi/sqrt[sum([(Wi*constraint)/l2_loss]^2)]
def l2_normalize(W, L2_NORM_CONSTRAINT):
    l2_loss = sqrt(2 * tf.nn.L2_loss(W))
    if  l2_loss > L2_NORM_CONSTRAINT:
        W = tf.scalar_mul(1/sqrt(tf.reduce_sum(tf.square(
            tf.scalar_mul(L2_NORM_CONSTRAINT/l2_loss, W), 2))), W)
    return W

def shuffle_file(params, lines, d):
    input_file = open(params['TRAIN_FILE_NAME'] + '.data', 'r')
    output_file = open(params['TRAIN_FILE_NAME'] + '.labels', 'r')
    input_list = []
    output_list = []
    for line in range(lines):
        input_list.append(line_to_vec(input_file.readline(), d, params))
        output_list.append(line_to_vec(output_file.readline(), d, params))
    z = zip(input_list, output_list)
    random.shuffle(z)
    return zip(*z)

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
    list_of_words = tokenize(clean_str(sample, params['SST']))
    word_vectors = []
    for word in list_of_words:
        word_vectors.extend(d[word])
    return word_vectors

#create a vocabulary list from a file
def find_vocab(file_name, SST, vocab=None, master_key=None):
    if vocab is None:
        vocab = []
    if master_key is None:
        master_key = {}
    text_file = open(file_name, 'r')
    list_of_words = tokenize(clean_str(text_file.read()), SST=SST)
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
        if line[0] in vocab:
            master_key[vocab[vocab.index(line[0])]] = line[1:]
            vocab.remove(line[0])
    for word in vocab:
        master_key[word] = [0] * 300
    return master_key

if __name__ == "__main__": main()

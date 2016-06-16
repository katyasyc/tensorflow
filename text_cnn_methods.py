import tensorflow as tf
import sys, re
import random
import numpy as np
import math
import os.path

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
    output.write('using batch size ' + str(params['BATCH_SIZE']))
    if params['USE_TFIDF']:
        output.write(', tfidf, ')
    else:
        output.write(', ')
    if params['USE_WORD2VEC']:
        output.write('word2vec, ')
    else:
        output.write('random init, ')
    if params['UPDATE_WORD_VECS']:
        output.write('updating.\n')
    else:
        output.write('not updating.\n')
    return output

#get random batch of examples from train file
def get_batches(params, train_x, train_y):
    if params['epoch'] == 1:
        np.random.seed(3435)
        if train_y.shape[0] % params['BATCH_SIZE'] > 0:
            extra_data_num = params['BATCH_SIZE'] - train_y.shape[0] % params['BATCH_SIZE']
            train_set_x, train_set_y = shuffle_in_unison(train_x, train_y)
            extra_data_x = train_set_x[:extra_data_num]
            extra_data_y = train_set_y[:extra_data_num]
            new_data_x = np.append(train_x, extra_data_x, axis=0)
            new_data_y = np.append(train_y, extra_data_y, axis=0)
        else:
            new_data_x = train_x
            new_data_y = train_y
    new_data_x, new_data_y = shuffle_in_unison(new_data_x, new_data_y)
    return new_data_x, new_data_y

#index and loop through same batches again
def get_batch(batches_x, batches_y, index, params):
    cur_batch_x = batches_x[index*params['BATCH_SIZE']:(index+1)*params['BATCH_SIZE'],:]
    cur_batch_y = batches_y[index*params['BATCH_SIZE']:(index+1)*params['BATCH_SIZE'],:]
    # if params['USE_TFIDF']:
    #     cur_batch_tfidf = batches_tfidf[index*params['BATCH_SIZE']:(index+1)*params['BATCH_SIZE'],:]
    return cur_batch_x, cur_batch_y

#initializes weights, random with stddev of .1
def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

#initializes biases, all at .1
def bias_variable(shape):
      initial = tf.zeros(shape=shape)
      return tf.Variable(initial)

#defines the first two layers of our neural network
def define_nn(x, kernel_size, params, slices, weights, biases):
    W = weight_variable([kernel_size, 1, params['WORD_VECTOR_LENGTH'], params['FILTERS']])
    b = bias_variable([params['FILTERS']])
    #convolve: each neuron iterates by 1 filter, 1 word
    conv = tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding="SAME")
    #apply bias and relu
    relu = tf.nn.relu(tf.nn.bias_add(conv, b))
    #max pool; each neuron sees 1 filter and returns max over a sentence

    #FIX: PROBABLY BREAKS if BATCH_SIZE is odd!!!
    if params['BATCH_SIZE'] > 1:
        pooled = tf.nn.max_pool(relu, ksize=[1, params['MAX_LENGTH'], 1, 1],
            strides=[1, params['MAX_LENGTH'], 1, 1], padding='SAME')
        slices.insert(len(slices), pooled)
    else:
        slices.insert(len(slices), relu)
    weights.insert(len(weights), W)
    biases.insert(len(biases), b)
    return slices, weights, biases

def one_hot(category, CLASSES):
    one_hot = [0] * CLASSES
    one_hot[category] = 1
    return np.asarray(one_hot)

#pads all sentences to same length
def pad(list_of_words, params):
    left = (params['MAX_LENGTH']  - len(list_of_words)) / 2
    right = left
    if (params['MAX_LENGTH'] - len(list_of_words)) % 2 != 0:
        right += 1
    for i in range(left):
        list_of_words.insert(0, '<PAD>')
    for i in range(right):
        list_of_words.append('<PAD>')
    return list_of_words

#get all examples from a file and return np arrays w/input and output
def get_strings(directory, file_name, params):
    input_file = open(os.path.expanduser("~") + '/convnets/tensorflow/' + os.path.join(directory, file_name) + '.data', 'r')
    input_list = []
    for line in input_file:
        input_list.append(clean_str(line, params))
    return input_list

def get_max_length(directory, train_file, dev_file):
    train = open(os.path.expanduser("~") + '/convnets/tensorflow/' + os.path.join(directory, train_file) + '.data', 'r')
    dev = open(os.path.expanduser("~") + '/convnets/tensorflow/' + os.path.join(directory, dev_file) + '.data', 'r')
    max_length = 0
    for line in dev:
        list_of_words = tokenize(line)
        max_length = max(max_length, len(list_of_words))
    for line in train:
        list_of_words = tokenize(line)
        max_length = max(max_length, len(list_of_words))
    return max_length
#get all examples from a file and return np arrays w/input and output
def get_all(directory, file_name, params):
    input_file = open(os.path.expanduser("~") + '/convnets/tensorflow/' + os.path.join(directory, file_name) + '.data', 'r')
    output_file = open(os.path.expanduser("~") + '/convnets/tensorflow/' + os.path.join(directory, file_name) + '.labels', 'r')
    # input_file = open(os.path.join(os.path.expanduser("~") + '/repos/tensorflow/' + file_name) + '.data', 'r')
    # output_file = open(os.path.join(os.path.expanduser("~") + '/repos/tensorflow/' + file_name) + '.labels', 'r')
    input_list = []
    output_list = []
    for line in input_file:
        input_list.append(clean_str(line, params))
    for line in output_file:
        output_list.append(one_hot(int(line.rstrip()), params['CLASSES']))
    return input_list, output_list

def number_of_tokens(string1):
   list_of_words = tokenize(string1)
   return len(list_of_words)

#takes a batch of text, key with vocab indexed to vectors
#returns word vectors concatenated into a list
def sub_vectors(input_list, d, params):
    list_of_examples = []
    for i in range(len(input_list)):
        list_of_words = []
        for j in range(params['MAX_LENGTH']):
            list_of_words.append(d[input_list[i][j]])
        list_of_examples.append(list_of_words)
    return np.expand_dims(np.asarray(list_of_examples), 2)

def sort_examples_by_length(input_list, output_list):
    lengths = []
    for example in input_list:
        lengths.append(number_of_tokens(example))
    new_lengths = []
    new_input_list = []
    new_output_list = []
    for i in range(len(lengths)):
        for j in range(len(new_lengths)):
            if lengths[i] < new_lengths[j]:
                new_lengths.insert(j, lengths[i])
                new_input_list.insert(j, input_list[i])
                new_output_list.insert(j, output_list[i])
                break
        else:
            new_lengths.append(lengths[i])
            new_input_list.append(input_list[i])
            new_output_list.append(output_list[i])
    return new_input_list, new_output_list

#shuffle two numpy arrays in unison
def shuffle_in_unison(a, b):
    rng_state = np.random.get_state()
    np.random.shuffle(a)
    np.random.set_state(rng_state)
    np.random.shuffle(b)
    # np.random.set_state(rng_state)
    # np.random.shuffle(c)
    return a, b

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
def clean_str(string, params):
    if params['SST'] == True:
        """
        Tokenization/string cleaning for the SST dataset
        """
        string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
        string = re.sub(r"\s{2,}", " ", string)
        return string.strip().lower()
    if params['ICMB'] == True:
        string = string.replace('<br /><br />', ' ')
        string = re.sub(r"-", " - ", string)
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
    return string.strip() if params['TREC'] else string.strip().lower()

#create a vocabulary list from a file
def find_vocab(list_of_sentences, vocab=None, master_key=None):
    list_of_words = [word for sentence in list_of_sentences for word in sentence]
    if vocab is None:
        vocab = []
    if master_key is None:
        master_key = {}
    for word in list_of_words:
        if word not in master_key and word not in vocab:
            vocab.append(word)
    return vocab

#initialize dict of vocabulary with word2vec or random numbers
def initialize_vocab(vocab, params):
    embed_keys = {}
    key_list = []
    if params['USE_WORD2VEC']:
        word2vec = open(params['WORD_VECS_FILE_NAME'], 'r')
        word2vec.readline()
        for i in range(3000000):   #number of words in word2vec
            line = tokenize(word2vec.readline().strip())
            #turn into floats
            if line[0] in vocab:
                vector = []
                for j in range(1, len(line)):
                    vector.append(float(line[j]))
                key_list.append(vector)
                embed_keys[line[0]] = len(embed_keys)
                vocab.remove(line[0])
        word2vec.close()
    for word in vocab:
        if word == '<PAD>':
            key_list.append(np.zeros([params['WORD_VECTOR_LENGTH']]))
        else:
            key_list.append(np.random.uniform(-0.25,0.25,params['WORD_VECTOR_LENGTH']))
        embed_keys[word] = len(embed_keys)
    return embed_keys, key_list

if __name__ == "__main__": main()

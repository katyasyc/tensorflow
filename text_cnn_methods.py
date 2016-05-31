import tensorflow as tf

#initializes weights, random with stddev of .1
def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

#initializes biases, all at .1
def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

#defines the first two layers of our neural network
def define_nn(x, kernel_size, FILTERS, WORD_VECTOR_LENGTH):
    #define weights and biases, make sure we can specify to normalize later
    #correct line: getting error
    #2nd dimension should be "None"
    #fix kernel_size
    print kernel_size
    W = weight_variable([3, WORD_VECTOR_LENGTH, 1, FILTERS])
    b = bias_variable([FILTERS])
    #convolve: each neuron iterates by 1 filter, 1 word
    print tf.shape(x)
    print tf.size(x)
    conv = tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding="SAME")
    #apply bias and relu
    relu = tf.nn.relu(tf.nn.bias_add(conv, b))
    #max pool; each neuron sees 1 filter and returns max over l
    pooled = tf.nn.max_pool(relu, ksize=[1, None, 1, 1],
        strides=[1, None, 1, 1], padding='SAME')
    return pooled, W, b

def pad(batch_x, length, WORD_VECTOR_LENGTH):
    for sample in batch_x:
        left = (length - len(sample)) / 2
        right = left
        if (length - len(sample)) % 2 != 0:
            right += 1
        sample = sample.insert(0, [0] * WORD_VECTOR_LENGTH * left)
        sample = sample.extend([0] * WORD_VECTOR_LENGTH * right)
    return batch_x

#returns one less than the number of lines in a file
def find_lines(file_name):
    for i, l in enumerate(file_name):
        pass
    return i

#l2_loss = l2 loss (tf fn returns half of l2 loss w/o sqrt)
#where Wi is each item in W, W = Wi/sqrt[sum([(Wi*constraint)/l2_loss]^2)]
def l2_normalize(W, L2_NORM_CONSTRAINT):
    l2_loss = sqrt(2 * tf.nn.L2_loss(W))
    if  l2_loss > L2_NORM_CONSTRAINT:
        W = tf.scalar_mul(1/sqrt(tf.reduce_sum(tf.square(
            tf.scalar_mul(L2_NORM_CONSTRAINT/l2_loss, W), 2))), W)
    return W

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

#takes a line of text, key with vocab indexed to vectors
#returns word vectors concatenated into a list
def line_to_vec(data, d, WORD_VECTOR_LENGTH, padding):
    list_of_words = tokenize(data.lowercase())
    word_vectors = [0] * (WORD_VECTOR_LENGTH * padding)
    for word in list_of_words:
        word_vectors.extend(d[word])
    word_vectors.extend([0] * (WORD_VECTOR_LENGTH * padding))
    return word_vectors

#initialize vocabulary of file_name with word2vec or vecs initialized with zeroes
#taking out found (replacing it with querying d again) would make code cleaner, but slower
def initialize_vocab(d, file_name, vectors_vocab, word_vectors):
    text_file = open(file_name, 'r')
    word2vec_vocab = open(vectors_vocab, 'r')
    word2vec_vectors = open(word_vectors, 'r')
    words = tokenize(text_file.read())
    for word in words:
        if word not in d:
            for i in range(3000000):   #number of words in word2vec
                line = tokenize(word2vec_vocab.readline().strip())
                if word == line[0]:
                    word2vec_vectors.seek(int(line[1]))
                    d[word] = tokenize(word2vec_vectors.readline().strip())
                    break
            else:
                d[word] = [0] * 300
    return d

if __name__ == "__main__": main()
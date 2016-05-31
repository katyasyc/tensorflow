#takes a line of text, key as file indexing words to bytes in data file
#returns concatenated word vectors in a list
#words not in word2vec initialized as [0] * 300
#problem: has to regenerate key each time it runs a line

#takes a line of text, returns an array of strings where ecah string is a word
def to_list(line):
   list_of_words = []
   word = ''
   for char in line:
      if char == ' ':
         list_of_words.append(word)
         word = ''
      else:
         word += char
   list_of_words.append(word)
   return list_of_words

#creates a map from key_file
def map_file(key_file):
    keys = {}
    string1 = key_file.readline()
    while string1 != '':
        split = string1.index(' ')
        keys[string1[0:split]] = string1[split+1:len(string1)]
        string1 = key_file.readline()
    return keys

#creates a long vector to be reshaped into a tensor for input
def create_word_vectors(list_of_words, WORD_VECTOR_LENGTH, padding):
  data_file = open('data.txt', 'r')
  word_vectors = [0] * WORD_VECTOR_LENGTH * padding
  for i in range(len(list_of_words)):
      list_of_words[i] = list_of_words[i].strip()
      if list_of_words[i] in keys:
           data_file.seek(int(keys[list_of_words[i]]))
           word_vectors.append(to_list(data_file.readline().rstrip('\n')))
      else:
           word_vectors.extend([0] * WORD_VECTOR_LENGTH)
   word_vectors.extend([0] * WORD_VECTOR_LENGTH * padding)
   return word_vectors

def main(data, WORD_VECTOR_LENGTH, padding):
   #create key
   key_file = open('key.txt', 'r')
   keys = map_file(key_file)
   list_of_words = to_list(datalowercase())
   word_vectors = create_word_vectors(list_of_words, WORD_VECTOR_LENGTH, padding)
   return word_vectors

if __name__ == "__main__": main()

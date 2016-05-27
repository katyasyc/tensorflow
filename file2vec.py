#takes test file, key as file indexing words to bytes in data file
#analyzes test file, returning concatenated word vectors separated by : for each line of input
#words not in word2vec initialized as [0] * 300

import sys
from random import *

#turns a string (line of text stripped of punctuation) into an array of strings
#each string in the array "words" is a word
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

#lowercases strings, removes non-letter characters
def clean_up(string1):
   string1 = string1.lower()
   string1 = string1.replace('\n',' \n')
   string1 = string1.replace('  ',' ')
   string1 = string1.replace('\\','')
   return string1

def map_file(key_file):
    keys = {}
    string1 = key_file.readline()
    while string1 != '':
        split = string1.index(' ')
        keys[string1[0:split]] = string1[split+1:len(string1)]
        string1 = key_file.readline()
    return keys

def main():
   test_file = open('test.data', 'r')
   vector_file = open('vectors.data', 'w')
   key_file = open('key.txt', 'r')
   data_file = open('data.txt', 'r')
   keys = map_file(key_file)
   string1 = test_file.read()
   string1 = clean_up(string1)
   list_of_words = to_list(string1)
   #index of examples-so we can map them to labels
   example_index = 0
   vector_file.write(str(example_index) + ' ; ')
   for i in range(len(list_of_words)):
       if '\n' in list_of_words[i]:
           vector_file.write('\n')
           example_index += 1
           if i != len(list_of_words):
               vector_file.write(str(example_index) + ' ; ')
       list_of_words[i] = list_of_words[i].strip()
       if list_of_words[i] in keys:
            data_file.seek(int(keys[list_of_words[i]]))
            vector_file.write(data_file.readline().rstrip('\n') + ' : ')
       else:
            vector_file.write("0 " * 300 + ': ')

if __name__ == "__main__": main()

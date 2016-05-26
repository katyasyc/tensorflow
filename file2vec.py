#takes test file, key as file indexing words to bytes in data file,
#returns byte indices in data file of words in test file

import sys
from random import *

#turns a string (line of text stripped of punctuation) into an array of strings
#each string in the array "words" is a word
def to_list(line):
   list_of_words = []
   word = ''
   for i in range (0,len(line)):
      if line[i] == ' ':
         list_of_words.append(word)
         word = ''
      else:
         word += line[i]
   list_of_words.append(word)
   return list_of_words

#lowercases strings, removes non-letter characters
def clean_up(string1):
   string1 = string1.lower()
   string1 = string1.replace('\n',' \n')
   string1 = string1.replace('  ',' ')
   string1 = string1.replace('\\','')

   """
   fix this at later date, or never
   letters = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', ' ']

   for i in range (0,len(string1)):
      string1iletter = False
      for j in range (0,len(letters)):
         if string1[i] == letters[j]:
            string1iletter = True
      if string1iletter == False:
         string1 = string1.replace(string1[i],'')
   string1 = string1.replace('  ','')
   """
   return string1
"""
#turns a list of words into one-hot vectors
def find_vocab(words):
   vocab = []
   for i in range(0,len(words)):
      word = words[i]
      if words[i] not in vocab:
         vocab.append(word)
   return vocab

def vector(words,vocab,i):
   vector = [0] * len(words)
   vector[vocab.index(words[i])] = 1
   return vector
"""

def map_file(key_file):
    keys = {}
    string1 = key_file.readline()
    while string1 != '':
        split = string1.index(' ')
        keys[string1[0:split]] = string1[split+1:len(string1)]
        string1 = key_file.readline()
    return keys

def main():
#how do I import any file (specify in console rather than program?
   test_file = open('test.data', 'r')
   vector_file = open('vectors.data', 'w')
   key_file = open('key.txt', 'r')
   data_file = open('data.txt', 'r')
   keys = map_file(key_file)
   string1 = test_file.read()
   string1 = clean_up(string1)
   list_of_words = to_list(string1)
   for i in range (0,len(list_of_words)):
       if '\n' in list_of_words[i]:
           vector_file.write('\n')
       list_of_words[i] = list_of_words[i].strip()
       if list_of_words[i] in keys:
            data_file.seek(int(keys[list_of_words[i]]))
            vector_file.write(data_file.readline())
       else:
            vector_file.write("0 " * 299 + '0\n')


   """
   vocab = find_vocab(words)
   print vocab

   word_vectors = []
   for i in range(0,len(vocab)):
      word_vectors.append(vector(words,vocab,i))
"""
if __name__ == "__main__": main()

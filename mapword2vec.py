#to convert word2vec .txt to a dictionary, and then write that to a file

import sys
import os

def add(line, vocab_file, vectors_file):
    split = line.index(' ')
    vectors_file.seek(0, os.SEEK_END)
    filesize = vectors_file.tell()
    vocab_file.write(line[0:split] + ' ' + str(filesize) + '\n')
    vectors_file.write(line[split+1:len(line)])

def main():
   vocab_file = open('word2vec_vocab.txt', 'w')
   vectors_file = open('word2vec_vectors.txt', 'r+')
   vectors_file.truncate()
   input_file = open('output.txt', 'r')
   input_file.readline()    #get rid of first line, which states length
   line = input_file.readline()
   while line != '':
      add(line, vocab_file, vectors_file)
      line = input_file.readline()
   key_file.close()
   data_file.close()
   input_file.close()

if __name__ == "__main__": main()

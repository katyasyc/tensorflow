#to convert word2vec .txt to a dictionary, and then write that to a file

import sys
import os

def add(line,key_file,data_file):
    split = line.index(' ')
    data_file.seek(0, os.SEEK_END)
    filesize = data_file.tell()
    key_file.write(line[0:split] + ' ' + str(filesize) + '\n')
    data_file.write(line[split+1:len(line)])

def main():
   key_file = open('key.txt', 'w')
   data_file = open('data.txt', 'r+')
   data_file.truncate()
   input_file = open('output.txt', 'r')
   input_file.readline()
   line = input_file.readline()
   while line != '':
      add(line,key_file,data_file)
      line = input_file.readline()
   key_file.close()
   data_file.close()
   input_file.close()

if __name__ == "__main__": main()

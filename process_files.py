#takes the files that come with the CR database and exports them into a form that our convnet can use
#input file path of directory where reviews are kept
import os.path

def process_MR(file_path):
    with open(os.path.join(file_path, 'MR.data'), 'w') as data:
        with open(os.path.join(file_path, 'MR.labels'), 'w') as labels:
            with open(os.path.join(file_path, 'rt-polarity.neg'), 'r') as neg:
                for line in neg:
                    data.write(line)
                    labels.write('0\n')
            with open(os.path.join(file_path, 'rt-polarity.pos'), 'r') as pos:
                for line in pos:
                    data.write(line)
                    labels.write('1\n')

def process_subj(file_path):
    with open(os.path.join(file_path, 'CR.data'), 'w') as data:
        with open(os.path.join(file_path, 'CR.labels'), 'w') as labels:
            pass
            # with open()

def process_MPQA(file_path):
    with open(os.path.join(file_path, 'MPQA.data'), 'w') as data:
        with open(os.path.join(file_path, 'MPQA.labels'), 'w') as labels:
            #for folder in file_path:
                #for file in folder:
            pass

def separate_labels_TREC(f, data, labels):
    for line in f:
        start = line.index(' ')
        data.write(line[start + 1:])
        label = line.index(':')
        if line[:label] == 'ABBR':
            labels.write('0' + '\n')
        elif line[:label] == 'ENTY':
            labels.write('1' + '\n')
        elif line[:label] == 'DESC':
            labels.write('2' + '\n')
        elif line[:label] == 'HUM':
            labels.write('3' + '\n')
        elif line[:label] == 'LOC':
            labels.write('4' + '\n')
        elif line[:label] == 'NUM':
            labels.write('5' + '\n')
        else:
            print 'label not found'
            f.close()
            data.close()
            labels.close()
            sys.exit(2)

def process_TREC(file_path):
    with open(os.path.join(file_path, 'TREC.data'), 'w') as data:
        with open(os.path.join(file_path, 'TREC.labels'), 'w') as labels:
            with open(os.path.join(file_path, 'raw_TREC'), 'r') as train:
                separate_labels_TREC(train, data, labels)
            with open(os.path.join(file_path, 'raw_test_TREC'), 'r') as test:
                separate_labels_TREC(test, data, labels)

def process_CR(file_path):
    with open(os.path.join(file_path, 'CR.data'), 'w') as data:
        with open(os.path.join(file_path, 'CR.labels'), 'w') as labels:
            for file_name in ('Apex AD2600 Progressive-scan DVD player.txt',
                             'Canon G3.txt',
                             'Creative Labs Nomad Jukebox Zen Xtra 40GB.txt',
                             'Nikon coolpix 4300.txt',
                             'Nokia 6610.txt'):
                #with open_file(file_path, file_name) as f:
                #path = os.path.join(os.path.expanduser("~"), file_path)
                #with open(os.path.join(path, file_name), 'r') as f:
                with open(os.path.join(file_path, file_name), 'r') as f:
                    for line in f:
                        if '##' in line:
                            start = line.find('##')
                            if '[+' in line[:start] and line[:start].count('[-') == 0:
                                data.write(str(line[start + 2:len(line) - 2]) + '\n')
                                labels.write(str(1) + '\n')
                            elif '[-' in line[:start] and line[:start].count('[+') == 0:
                                data.write(str(line[start + 2:len(line) - 2]) + '\n')
                                labels.write(str(0) + '\n')

            """
            while feed != '' or examples == []:
                if feed.count('[t]') > 1:
                    start_example = feed.index('[t]')
                    del feed[:start_example + 3]
                    end_example = feed.index('[t]')
                    examples.append(feed[:end_example] + '\n')
                elif next_line == '' and feed.count('[t]') == 1:
                    start_example = feed.index('[t]')
                    examples.append(feed[start_example:])
                else:
                    feed.append(next_line)
                    next_line = f.readline()
            print examples
            for line in range(lines):
                input_list.append(pad(tokenize(clean_str(input_file.readline(), SST = params['SST'])), params))
                output_list.append(one_hot(int(output_file.readline().rstrip()), params['CLASSES']))"""

if __name__ == "__main__": main()

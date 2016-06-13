import sys
def main(args):
    for arg in args:
        fi = arg
        f = open(fi)
        output = []
        for line in f:
            if 'clipped' in line:
                pass
            elif 'numpy' in line:
                pass
            else:
                output.append(line)
        f.close()
        f = open(fi, 'w')
        f.writelines(output)
        f.close()

if __name__ == "__main__": main(sys.argv[1:])

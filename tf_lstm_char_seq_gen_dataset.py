import random


# gen 1000 insrtances of 30 works
input = open('char_seq/data.txt', 'r').read().split('X:')
for i in range(1, 1000):
    print("X:" + input[random.randint(1, 30)] + \
    "\n____________________________________\n")

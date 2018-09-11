import numpy as np


def char2int(char):
    if(char == ' '):
        return 27
    return ord(char)-ord('a')+1
    
def int2char(num):
    if(num == 27):
        return ' '
    return chr(ord('a')+num-1)

def word_to_vector(word):
    vector = []
    for c in list(word):
        vector.append(char2int(c))
    return vector

def vector_to_word(vector):
    word = ""
    for x in vector:
        word = word + int2char(x)
    return word
    

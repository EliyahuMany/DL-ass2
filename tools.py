import torch
from itertools import izip
import numpy as np

WORDS = set()
TAGS = set()
NEW_LINE = '\n'
SIZE = 50
START = '*START*'
END = '*END*'
UNK = "UUUNKKK"
W2I = {}
I2W = {}
T2I = {}
I2T = {}


def creates_dicts(words, tags):
    global W2I, I2W, T2I, I2T
    words.update(set([START, END]))
    W2I = {word: i for i, word in enumerate(words)}
    I2W = {i: word for word, i in W2I.iteritems()}
    T2I = {tag: i for i, tag in enumerate(tags)}
    I2T = {i: tag for tag, i in T2I.iteritems()}


def word_index(word):
    if word in W2I:
        return W2I[word]
    else:
        return W2I[UNK]


def indices_window(word_1, word_2, word_3, word_4, word_5):
    win = []
    win.append(word_index(word_1))
    win.append(word_index(word_2))
    win.append(word_index(word_3))
    win.append(word_index(word_4))
    win.append(word_index(word_5))
    return win


def windows_and_tags(sentences):
    words = []
    tags = []
    for s in sentences:
        pad = [(START, START), (START, START)]
        pad.extend(s)
        pad.extend([(END, END), (END, END)])
        for i, (word, tag) in enumerate(pad):
            if word is not START and word is not END:
                words_pack = [pad[i - 2][0], pad[i - 1][0], word, pad[i + 1][0], pad[i + 2][0]]
                words.append(indices_window(words_pack[0], words_pack[1], words_pack[2], words_pack[3], words_pack[4]))
                tags.append(T2I[tag])
    return words, tags


def windows(sentences):
    concat_words = []
    for sentence in sentences:
        pad = [START, START]
        pad.extend(sentence)
        pad.extend([END, END])
        for i, (word) in enumerate(pad):
            if word is not START and word is not END:
                words_pack = [pad[i - 2], pad[i - 1], word, pad[i + 1], pad[i + 2]]
                concat_words.append(
                    indices_window(words_pack[0], words_pack[1], words_pack[2], words_pack[3], words_pack[4]))
    return concat_words


def get_tagged(file_name, dev):
    global WORDS, TAGS
    sentences_tagged = []
    with open(file_name) as f:
        content = f.readlines()
        sentence_and_tags = []
        for line in content:
            if line is NEW_LINE:
                sentences_tagged.append(sentence_and_tags)
                sentence_and_tags = []
                continue
            line = line.strip(NEW_LINE).strip().strip("\t")
            word, tag = line.split()
            if not dev:
                TAGS.add(tag)
                WORDS.add(word)
            sentence_and_tags.append((word, tag))
            # if line is not NEW_LINE:
            #     line = line.strip(NEW_LINE).strip().strip("\t")
            #     word, tag = line.split()
            #     if not dev:
            #         TAGS.add(tag)
            #         WORDS.add(word)
            #     sentence_and_tags.append((word, tag))
            # else:
            #     sentences_tagged.append(sentence_and_tags)
            #     sentence_and_tags = []
    if not dev:
        TAGS.add(UNK)
        WORDS.add(UNK)
        creates_dicts(WORDS, TAGS)
    return windows_and_tags(sentences_tagged)


def not_tagged(file_name):
    global WORDS, TAGS
    sentences = []
    with open(file_name) as file:
        content = file.readlines()
        sentence = []
        for line in content:
            if line == "\n":
                sentences.append(sentence)
                sentence = []
                continue
            w = line.strip("\n").strip()
            sentence.append(w)
    return windows(sentences)


def word_dict(words, vector):
    word_dict = {}
    for word, vector in izip(open(words), open(vector)):
        word = word.strip(NEW_LINE).strip()
        vector = vector.strip(NEW_LINE).strip().split(" ")
        word_dict[word] = np.asanyarray(map(float, vector))
    return word_dict

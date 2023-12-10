import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from bidi.algorithm import get_display
import arabic_reshaper
import string
import re
import os
import numpy as np
import pandas as pd

# define constants
DIACRITIC_NAMES = ['Fathatan', 'Dammatan', 'Kasratan', 'Fatha', 'Damma', 'Kasra', 'Shadda', 'Sukun']
NAME2DIACRITIC = dict((name, chr(code)) for name, code in zip(DIACRITIC_NAMES, range(0x064B, 0x0653)))
DIACRITIC2NAME = dict((code, name) for name, code in NAME2DIACRITIC.items())
ARABIC_DIACRITICS = frozenset(NAME2DIACRITIC.values())
ARABIC_LETTERS = frozenset([chr(x) for x in (list(range(0x0621, 0x63B)) + list(range(0x0641, 0x064B)))])
ARABIC_SYMBOLS = ARABIC_LETTERS | ARABIC_DIACRITICS

EXTRA_SUKUN_REGEXP = re.compile(r'(?<=ال)' + NAME2DIACRITIC['Sukun'])

# YA_REGEXP = re.compile(r'ى(?=['+''.join(ARABIC_DIACRITICS)+r'])')
DIACRITIC_SHADDA_REGEXP = re.compile('(['+''.join(ARABIC_DIACRITICS)+'])('+NAME2DIACRITIC['Shadda']+')')
XML_TAG = r'(?:<.+?>)'
SENTENCE_SEPARATORS = ';,،؛.:؟!'
SPACES = ' \t'
PUNCTUATION = SENTENCE_SEPARATORS + '۩﴿﴾«»ـ' +\
              ''.join([chr(x) for x in range(0x0021, 0x0030)]+[chr(x) for x in range(0x003A, 0x0040)] +
                      [chr(x) for x in range(0x005B, 0x0060)]+[chr(x) for x in range(0x007B, 0x007F)])

# SPACE_PUNCTUATION_REGEXP = re.compile('[' + SPACES + PUNCTUATION + ']+')
ARABIC_DIACRITICS_REGEXP = re.compile('[^' + ''.join(ARABIC_DIACRITICS)+''.join(ARABIC_LETTERS) + '\s]')
DATETIME_REGEXP = re.compile(r'(?:\d+[-/:\s]+)+\d+')
NUMBER_REGEXP = re.compile(r'\d+(?:\.\d+)?')
ZERO_REGEXP = re.compile(r'\b0\b')
WORD_TOKENIZATION_REGEXP = re.compile(
    '((?:[' + ''.join(ARABIC_LETTERS) + ']['+''.join(ARABIC_DIACRITICS)+r']*)+|\d+(?:\.\d+)?)')
SENTENCE_TOKENIZATION_REGEXP = re.compile(r'([' + SENTENCE_SEPARATORS + r'])(?!\w)|' + XML_TAG)
CHAR2INDEX = dict((l, n) for n, l in enumerate(sorted(ARABIC_LETTERS)))
CHAR2INDEX.update(dict((v, k) for k, v in enumerate([' ', '0'], len(CHAR2INDEX))))
INDEX2CHAR = dict((v, k) for k, v in CHAR2INDEX.items())
DIACRITIC2INDEX = dict((l, n) for n, l in enumerate(sorted(ARABIC_DIACRITICS)))
DIACRITIC2INDEX.update(dict((v, k) for k, v in enumerate([''], len(DIACRITIC2INDEX))))
INDEX2DIACRITIC = dict((v, k) for k, v in DIACRITIC2INDEX.items())

# print arabic diacritics
def printDiacritics():
    for name, code in zip(DIACRITIC_NAMES, range(0x064B, 0x0653)):
        print(name, chr(code))

# load train and test data text
def loadText():
    # load train text
    train_text = []
    with open('dataset/train.txt','rt', encoding='utf-8') as f:
        train_text_line = f.readline().strip()
        # train_text.append(train_text_line)
        while(train_text_line != ""):
            train_text.append(train_text_line)
            train_text_line = f.readline().strip()

    # load test text
    test_text = []
    with open('dataset/val.txt', encoding='utf-8') as f:
        test_text_line = f.readline().strip()
        while test_text_line != "":
            test_text.append(test_text_line)
            test_text_line = f.readline().strip()
    return train_text, test_text


# print arabic text
def printFirstLine(line):
    # print first line of arabic text correctly
    # reshape arabic letters
    # line = arabic_reshaper.reshape(line)
    # print arabic letters correctly
    # print(line)
    line = get_display(line)
    # line = arabic_reshaper.reshape(line)
    print(line)


def clear_diacritics(text):
    assert isinstance(text, str)
    return ''.join([l for l in text if l not in ARABIC_DIACRITICS])

def extract_diacritics(text):
    assert isinstance(text, str)
    diacritics_list = []
    # split text to words on spaces
    words = text.split()
    # join words to one string
    text = ' '.join(words)
    for i in range(len(text)):
        # check if the character is a diacritic
        if text[i] in ARABIC_DIACRITICS:
            diacritics_list.append(text[i])
        # check if the character is a letter and the previous character is a letter so there is no diacritic
        elif i!=0 and text[i-1] in ARABIC_LETTERS:
            diacritics_list.append('')
    # check if the last character is a letter so there is no diacritic
    if text[-1] in ARABIC_LETTERS:
        diacritics_list.append('')
    return diacritics_list

def extract_diacritics_with_shadda(text):
    assert isinstance(text, str)
    diacritics_list = []
    for i in range(len(text)):
        # check if the character is a diacritic
        if text[i] in ARABIC_DIACRITICS:
            # check if previous character is shadda
            if text[i-1] == NAME2DIACRITIC['Shadda']:
                diacritics_list[-1] = (text[i-1], text[i])
            else:
                diacritics_list.append(text[i])
        # check if the character is a letter and the previous character is a letter so there is no diacritic
        elif text[i-1] in ARABIC_LETTERS:
            diacritics_list.append('')
    # check if the last character is a letter so there is no diacritic
    if text[-1] in ARABIC_LETTERS:
        diacritics_list.append('')
    return diacritics_list

def extract_diacritics_with_previous_letter(text):
    assert isinstance(text, str)
    diacritics_list = []
    i = 0
    while i <len(text):
        # check if the character is a arabic letter
        if text[i] in ARABIC_LETTERS:
            # check if next character is diacritic not shadda
            if i+1 < len(text):
                if text[i+1] in ARABIC_DIACRITICS - {NAME2DIACRITIC['Shadda']}:
                    diacritics_list.append([text[i], text[i+1]])
                    i += 1
                elif text[i+1] == NAME2DIACRITIC['Shadda'] and i+2< len(text) and \
                      text[i+2] in ARABIC_DIACRITICS - {NAME2DIACRITIC['Shadda']} :
                    diacritics_list.append([text[i], text[i+1], text[i+2]])
                    i += 1
                elif text[i+1] == NAME2DIACRITIC['Shadda']:
                    diacritics_list.append([text[i], text[i+1]])
                    i += 1
                else:
                    diacritics_list.append([text[i], ''])
                i+=1    
            else:
                diacritics_list.append([text[i], ''])
                i+=1
        # elif text[i] is ' ':
        #     diacritics_list.append([' ', ''])
        #     i+=1
        else:
            i+=1
    return diacritics_list


                

def merge_text_with_diacritics(undiacritized_text, diacritics):

    assert isinstance(undiacritized_text, str)
    assert set(diacritics).issubset(ARABIC_DIACRITICS.union(['']))
    i = 0
    j = 0
    sequence = []
    while i < len(undiacritized_text) and j < len(diacritics):
        # if the character is a letter append it to the sequence
        sequence.append(undiacritized_text[i])
        i += 1

        if diacritics[j] in ARABIC_DIACRITICS:
            sequence.append(diacritics[j])
            # check if the diacritic is shadda and the next diacritic is not shadda
            if DIACRITIC2NAME[diacritics[j]] == 'Shadda' and j+1 < len(diacritics) and \
                    diacritics[j+1] in ARABIC_DIACRITICS - {diacritics[j]}:
                sequence.append(diacritics[j+1])
                j += 1
        j += 1
    return ''.join(sequence)

def merge_tokenized_with_diacritics(tokenized_sentence, diacritics):
    assert isinstance(tokenized_sentence, list) and all(isinstance(w, str) for w in tokenized_sentence)
    # assert set(diacritics).issubset(ARABIC_DIACRITICS.union(['']))
    
    # diacritics is list of diacritics for sentence
    # tokenized_sentence is list of words in sentence
    # merge diacritics with words
    i = 0
    j = 0
    sequence = []
    while i < len(tokenized_sentence) and j < len(diacritics):
        # loop on each word in sentence
        word = tokenized_sentence[i]
        # loop on each character in word
        for char in word:
            sequence.append(char)

            if j < len(diacritics) and  diacritics[j] in ARABIC_DIACRITICS:
                sequence.append(diacritics[j])
                # check if the diacritic is shadda and the next diacritic is not shadda
                if DIACRITIC2NAME[diacritics[j]] == 'Shadda' and j+1 < len(diacritics) and \
                        diacritics[j+1] in ARABIC_DIACRITICS - {diacritics[j]}:
                    sequence.append(diacritics[j+1])
                    j += 1
            j += 1
        i += 1
        # add space after each word
        sequence.append(' ')
    return ''.join(sequence)
def merge_tokenized_with_diacritics2(tokenized_sentence, diacritics):
    assert isinstance(tokenized_sentence, list) and all(isinstance(w, str) for w in tokenized_sentence)
    # assert set(diacritics).issubset(ARABIC_DIACRITICS.union(['']))
    
    # diacritics is list of diacritics for sentence
    # tokenized_sentence is list of words in sentence
    # merge diacritics with words
    i = 0
    j = 0
    sequence = []
    while i < len(tokenized_sentence) and j < len(diacritics):
        # loop on each word in sentence
        word = tokenized_sentence[i]
        # loop on each character in word
        for char in word:
            sequence.append(char)

            if j < len(diacritics) and  diacritics[j] in ARABIC_DIACRITICS:
                sequence.append(diacritics[j])
                # check if the diacritic is shadda and the next diacritic is not shadda
                if DIACRITIC2NAME[diacritics[j]] == 'Shadda' and j+1 < len(diacritics) and \
                        diacritics[j+1] in ARABIC_DIACRITICS - {diacritics[j]}:
                    sequence.append(diacritics[j+1])
                    j += 1
            # check if diacritics[j] is tuple of shadda and diacritic
            elif j < len(diacritics) and isinstance(diacritics[j], tuple):
                sequence.append(diacritics[j][0])
                sequence.append(diacritics[j][1])
                j += 1
            j += 1
        i += 1
        # add space after each word
        sequence.append(' ')
    return ''.join(sequence)

def merge_tokenized_with_diacritics3(tokenized_sentence, diacritics):
    assert isinstance(tokenized_sentence, list) and all(isinstance(w, str) for w in tokenized_sentence)
    # assert set(diacritics).issubset(ARABIC_DIACRITICS.union(['']))
    
    # diacritics is list of diacritics for sentence
    # tokenized_sentence is list of words in sentence
    # merge diacritics with words
    i = 0
    j = 0
    sequence = []
    while i < len(tokenized_sentence) and j < len(diacritics):
        # loop on each word in sentence
        word = tokenized_sentence[i]
        # loop on each character in word
        for char in word:
            sequence.append(char)
            # check char is equal to diacritics[j][0]
            if char != diacritics[j][0]:
                print("Error: char is not equal to diacritics[j][0]")
                print("char: ",char)
                print("diacritics[j][0]: ",diacritics[j][0])
                exit()
            
            for diacritic_index in range(1,len(diacritics[j])):
                sequence.append(diacritics[j][diacritic_index])
            j += 1
            
        i += 1
        # add space after each word
        sequence.append(' ')
    return ''.join(sequence)



def clean_text(text):

    assert isinstance(text, str)
    # Clean HTML garbage, tatweel, dates.
    return DATETIME_REGEXP.sub('', text.replace('ـ', '').replace('&quot;', ''))

def fix_diacritics_errors(diacritized_text):
    assert isinstance(diacritized_text, str)
    # Remove the extra Sukun from ال
    diacritized_text = EXTRA_SUKUN_REGEXP.sub('', diacritized_text)
    # Fix misplaced Fathatan
    diacritized_text = diacritized_text.replace('اً', 'ًا')
    # Fix reversed Shadda-Diacritic
    diacritized_text = DIACRITIC_SHADDA_REGEXP.sub(r'\2\1', diacritized_text)
    # Fix ى that should be ي (disabled)
    # diacritized_text = YA_REGEXP.sub('ي', diacritized_text)
    # Remove the duplicated diacritics by leaving the second one only when there are two incompatible diacritics
    fixed_text = diacritized_text[0]
    for x in diacritized_text[1:]:
        if x in ARABIC_DIACRITICS and fixed_text[-1] in ARABIC_DIACRITICS:
            if fixed_text[-1] != NAME2DIACRITIC['Shadda'] or x == NAME2DIACRITIC['Shadda']:
                fixed_text = fixed_text[:-1]
        # Remove the diacritics that are without letters
        elif x in ARABIC_DIACRITICS and fixed_text[-1] not in ARABIC_LETTERS:
            continue
        fixed_text += x
    return fixed_text

def sentence_tokenizer(text,debug = False):
    sentences_splits = re.split(SENTENCE_TOKENIZATION_REGEXP, text)
    sentences = []
    for sentence in sentences_splits:
        if sentence is not None:
            if sentence.strip(SPACES) != '':
                sentences.append(sentence)
    if debug:
        for sentence in sentences:
            print("sentence: ",sentence)
    return sentences


def tokenize(sentence):
    assert isinstance(sentence, str)
    return list(filter(lambda x: x != '' and x.isprintable(), re.split(WORD_TOKENIZATION_REGEXP, sentence)))

def filter_tokenized_sentence(sentence, min_words=1, min_word_diac_rate=0.8, min_word_diac_ratio=0.5):
    assert isinstance(sentence, list) and all(isinstance(w, str) for w in sentence)
    assert min_words >= 0
    assert min_word_diac_rate >= 0
    new_sentence = []
    if len(sentence) > 0:
        diac_word_count = 0
        arabic_word_count = 0
        for token in sentence:
            token = token.strip(SPACES)
            if not token:
                continue
            word_chars = set(token)
            if word_chars & ARABIC_LETTERS != set():
                arabic_word_count += 1
                word_diacs = extract_diacritics_with_shadda(token)
                if len([x for x in word_diacs if x]) / len(word_diacs) >= min_word_diac_ratio:
                    diac_word_count += 1
            new_sentence.append(token)
        if arabic_word_count > 0 and arabic_word_count >= min_words:
            if diac_word_count / arabic_word_count >= min_word_diac_rate:
                return new_sentence
    return []


def remove_non_arabic_chars(text):
    assert isinstance(text, str)
    # using regex
    line = re.sub(ARABIC_DIACRITICS_REGEXP,'',text)
    # remove more than 2 space between words
    line = re.sub(' +', ' ', line)
    return line

def preprocessing_text(text,name,debug = False):
    # tokenize text to sentences
    sentences = []
    # loop on each line "document"
    for i in range(len(text)):
        # get line
        line = text[i]

        # clean line from non arabic characters
        line = clean_text(line)

        # try nltk sentence tokenizer
        line_sentences = nltk.sent_tokenize(line)
        if debug:
            print("line_sentences",line_sentences)
        # tokenize line to sentences using tokenizer or regex
        line_sentences = sentence_tokenizer(line)


        if len(line_sentences) > 1:
            for i in range(0,len(line_sentences),2):
                if i+1 < len(line_sentences):
                    sentences.append(line_sentences[i]+line_sentences[i+1])
                else:
                    sentences.append(line_sentences[i])
        else:
            sentences.extend(line_sentences)

    filtered_sentences = []
    # loop on each sentence
    for i in range(len(sentences)):
        # filter sentence
        possible_sentences = filter_tokenized_sentence(tokenize(fix_diacritics_errors(sentences[i])))
        if len(possible_sentences) > 0:
            filtered_sentences.append(' '.join(possible_sentences))

    # save filtered sentences in text file
    with open("dataset/"+name, 'w', encoding='utf-8') as f:
        for sentence in filtered_sentences:
            f.write(remove_non_arabic_chars( sentence).strip()+'\n')
    with open("dataset/undiacritized_"+name, 'w', encoding='utf-8') as f:
        for sentence in filtered_sentences:
            f.write(clear_diacritics(remove_non_arabic_chars (sentence)).strip()+'\n')
    return filtered_sentences
            

    
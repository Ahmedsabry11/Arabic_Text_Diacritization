import re
import os
import numpy as np
import pandas as pd


class DataPreprocessing:
    # define constants in class to be used in all functions
    DIACRITIC_NAMES = ['Fathatan', 'Dammatan', 'Kasratan', 'Fatha', 'Damma', 'Kasra', 'Shadda', 'Sukun']
    NAME2DIACRITIC = dict((name, chr(code)) for name, code in zip(DIACRITIC_NAMES, range(0x064B, 0x0653)))
    # append shadda to diacritics
    NAME2DIACRITIC['Shadda_Dammatan'] = NAME2DIACRITIC['Shadda'] + NAME2DIACRITIC['Dammatan']
    NAME2DIACRITIC['Shadda_Kasratan'] = NAME2DIACRITIC['Shadda'] + NAME2DIACRITIC['Kasratan']
    NAME2DIACRITIC['Shadda_Fatha'] = NAME2DIACRITIC['Shadda'] + NAME2DIACRITIC['Fatha']
    NAME2DIACRITIC['Shadda_Damma'] = NAME2DIACRITIC['Shadda'] + NAME2DIACRITIC['Damma']
    NAME2DIACRITIC['Shadda_Kasra'] = NAME2DIACRITIC['Shadda'] + NAME2DIACRITIC['Kasra']
    NAME2DIACRITIC['Shadda_Fathatan'] = NAME2DIACRITIC['Shadda'] + NAME2DIACRITIC['Fathatan']

    DIACRITIC2NAME = dict((code, name) for name, code in NAME2DIACRITIC.items())
    ARABIC_DIACRITICS = frozenset(NAME2DIACRITIC.values())
    ARABIC_LETTERS = frozenset([chr(x) for x in (list(range(0x0621, 0x63B)) + list(range(0x0641, 0x064B)))])
    ARABIC_SYMBOLS = ARABIC_LETTERS | ARABIC_DIACRITICS

    SENTENCE_SEPARATORS = ';,،؛.:؟!'
    SENTENCE_TOKENIZATION_REGEXP = re.compile(r'([' + SENTENCE_SEPARATORS + r'])(?!\w)')
    # define SENTENCE_SEPARATORS as a set of characters
    SENTENCE_SEPARATORS = set(SENTENCE_SEPARATORS)
    ARABIC_SYMBOLS = ARABIC_SYMBOLS | SENTENCE_SEPARATORS
    # add space to arabic symbols
    ARABIC_SYMBOLS = ARABIC_SYMBOLS | {' '}

    SPACES = ' \t'
    EXTRA_SUKUN_REGEXP = re.compile(r'(?<=ال)' + NAME2DIACRITIC['Sukun'])

    # YA_REGEXP = re.compile(r'ى(?=['+''.join(ARABIC_DIACRITICS)+r'])')
    DIACRITIC_SHADDA_REGEXP = re.compile('(['+''.join(ARABIC_DIACRITICS)+'])('+NAME2DIACRITIC['Shadda']+')')

    CHAR2INDEX = dict((l, n) for n, l in enumerate(sorted(ARABIC_LETTERS)))
    CHAR2INDEX.update(dict((v, k) for k, v in enumerate([' ', '0','s'], len(CHAR2INDEX))))
    INDEX2CHAR = dict((v, k) for k, v in CHAR2INDEX.items())

    DIACRITIC2INDEX = dict((l, n) for n, l in enumerate(sorted(ARABIC_DIACRITICS)))
    DIACRITIC2INDEX.update(dict((v, k) for k, v in enumerate(['','0'], len(DIACRITIC2INDEX))))
    INDEX2DIACRITIC = dict((v, k) for k, v in DIACRITIC2INDEX.items())
    def __init__(self):
        pass
    def load_text(self,file_path):
        # read text file sentence by sentence
        sentences = []
        with open(file_path,'rt', encoding='utf-8') as f:
            text_line = f.readline().strip()
            # train_text.append(train_text_line)
            while(text_line != ""):
                sentences.append(text_line)
                text_line = f.readline().strip()
        return sentences
    def clean(self,text: str):
        """
        Clean text from non Arabic characters
        :param text: input text
        :return: cleaned text
        """
        assert isinstance(text, str)
        line =  ''.join([c for c in text if c in DataPreprocessing.ARABIC_SYMBOLS])
        line = self.fix_errors(line)
        # remove extra spaces
        line = re.sub(r'\s+', ' ', line)
        return line
    def fix_errors(self,diacritized_text):
        assert isinstance(diacritized_text, str)
        # Remove the extra Sukun from ال
        diacritized_text = DataPreprocessing.EXTRA_SUKUN_REGEXP.sub('', diacritized_text)
        # Fix misplaced Fathatan
        diacritized_text = diacritized_text.replace('اً', 'ًا')
        # Fix reversed Shadda-Diacritic
        diacritized_text = DataPreprocessing.DIACRITIC_SHADDA_REGEXP.sub(r'\2\1', diacritized_text)
        # Fix ى that should be ي (disabled)
        # diacritized_text = YA_REGEXP.sub('ي', diacritized_text)
        # Remove the duplicated diacritics by leaving the second one only when there are two incompatible diacritics
        fixed_text = diacritized_text[0]
        for x in diacritized_text[1:]:
            if x in DataPreprocessing.ARABIC_DIACRITICS and fixed_text[-1] in DataPreprocessing.ARABIC_DIACRITICS:
                if fixed_text[-1] != DataPreprocessing.NAME2DIACRITIC['Shadda'] or x == DataPreprocessing.NAME2DIACRITIC['Shadda']:
                    fixed_text = fixed_text[:-1]
            # Remove the diacritics that are without letters
            elif x in DataPreprocessing.ARABIC_DIACRITICS and fixed_text[-1] not in DataPreprocessing.ARABIC_LETTERS:
                continue
            fixed_text += x
        return fixed_text

    def extract_diacritics_with_previous_letter(self,text):
        assert isinstance(text, str)
        diacritics_list = []
        i = 0
        sentence = ''
        while i <len(text):
            # check if the character is a arabic letter
            if text[i] in DataPreprocessing.ARABIC_LETTERS:
                # check if next character is diacritic not shadda
                # add arabic letter to sentence
                sentence += text[i]
                if i+1 < len(text):
                    if text[i+1] in DataPreprocessing.ARABIC_DIACRITICS - {DataPreprocessing.NAME2DIACRITIC['Shadda']}:
                        diacritics_list.append(text[i+1])
                        i += 1
                    elif text[i+1] == DataPreprocessing.NAME2DIACRITIC['Shadda'] and i+2< len(text) and \
                        text[i+2] in DataPreprocessing.ARABIC_DIACRITICS - {DataPreprocessing.NAME2DIACRITIC['Shadda']} :
                        diacritics_list.append(text[i+1]+text[i+2])
                        i += 2
                    elif text[i+1] == DataPreprocessing.NAME2DIACRITIC['Shadda']:
                        diacritics_list.append(text[i+1])
                        i += 1
                    else:
                        diacritics_list.append('')
                    i+=1    
                else:
                    diacritics_list.append('')
                    i+=1
            elif text[i] == ' ':
                sentence += text[i]
                diacritics_list.append('')
                i+=1
            else:
                i+=1
        return diacritics_list,sentence
    
    def remove_punctuation(self,text):
        assert isinstance(text, str)
        line =  ''.join([l for l in text if l not in DataPreprocessing.SENTENCE_SEPARATORS])

        # remove extra spaces
        line = re.sub(r'\s+', ' ', line)
        return line
    def remove_non_arabic_chars(self,text):
        assert isinstance(text, str)
        line =  ''.join([c for c in text if c in DataPreprocessing.ARABIC_LETTERS|{' '}])

        # remove extra spaces
        line = re.sub(r'\s+', ' ', line)
        return line
    def sentence_tokenizer(self,text,debug = False):
        # first split document to sentences on end of sentence characters on
        sentences_splits =  re.split(r'\n', text)
        
        # loop on each sentence
        sentences = []
        for sentence in sentences_splits:
            if sentence is not None:
                # check if sentence is larger than 600 characters
                if len(sentence) > 600:
                    # split sentence on ;,،؛.:؟!
                    sentences_splits2 = re.split(DataPreprocessing.SENTENCE_TOKENIZATION_REGEXP, sentence)
                    for sentence2 in sentences_splits2:
                        if sentence2 is not None:
                            if sentence2.strip(DataPreprocessing.SPACES) != '':
                                if len(sentence2) > 50:
                                    sentences.append(sentence2)
                else:
                    if sentence.strip(DataPreprocessing.SPACES) != '':
                        if len(sentence) > 50:
                            sentences.append(sentence)
        return sentences

    def clear_diacritics(self,text: str):
        assert isinstance(text, str)
        return ''.join([l for l in text if l not in DataPreprocessing.ARABIC_DIACRITICS])

    def text_preprocessing(self,text,name,debug = False):
        # tokenize text to sentences
        sentences = []
        # loop on each line "document"
        for i in range(len(text)):
            # get line
            line = text[i]

            # clean line from non arabic characters
            line = self.clean(line)

            # tokenize line to sentences using tokenizer or regex
            line_sentences = self.sentence_tokenizer(line)

            if len(line_sentences) > 1:
                for i in range(0,len(line_sentences),2):
                    if i+1 < len(line_sentences):
                        sentences.append(line_sentences[i]+line_sentences[i+1])
                    else:
                        sentences.append(line_sentences[i])
            else:
                sentences.extend(line_sentences)

        filtered_sentences = sentences
        # loop on each sentence

        # for i in range(len(sentences)):
        #     # filter sentence
        #     possible_sentences = filter_tokenized_sentence(tokenize(fix_diacritics_errors(sentences[i])))
        #     if len(possible_sentences) > 0:
        #         filtered_sentences.append(' '.join(possible_sentences))

        # save filtered sentences in text file
        with open("dataset/"+name, 'w', encoding='utf-8') as f:
            for sentence in filtered_sentences:
                f.write(self.remove_punctuation(sentence).strip()+'\n')
        with open("dataset/undiacritized_"+name, 'w', encoding='utf-8') as f:
            for sentence in filtered_sentences:
                f.write(self.clear_diacritics( self.remove_punctuation(sentence)).strip()+'\n')

        return filtered_sentences
    def convert_char_to_vector(self,char):
        # convert each character to vector of 1s and 0s
        # get character index from dictionary
        index = DataPreprocessing.CHAR2INDEX[char]
        # create vector of zeros
        vector = [0] * len(DataPreprocessing.CHAR2INDEX)
        # set index to 1
        vector[index] = 1
        return np.array(vector)

    def convert_labels_to_indices(self,labels):
        # convert each character to vector of 1s and 0s
        # get character index from dictionary
        labels_vector = []
        for label in labels:
            label_vec = DataPreprocessing.DIACRITIC2INDEX[label]
            labels_vector.append(label_vec)
        labels_vector = np.array(labels_vector)
        # print("labels_vector",labels_vector.shape)
        return labels_vector
    def convert_sentence_to_indices(self,sentence):
        sentence_vector = []
        for char in sentence:
            char_vec = DataPreprocessing.CHAR2INDEX[char]
            sentence_vector.append(char_vec)
        return sentence_vector
    
    def convert_tokenized_sentence_to_vector(self,tokenzied_sentence):
        # convert each character to vector of 1s and 0s
        # get character index from dictionary
        sentence_vector = []
        for word in tokenzied_sentence:
            for char in word:
                char_vec = self.convert_char_to_vector(char)
                sentence_vector.append(char_vec)
        return np.array(sentence_vector)
    def convert_sentence_to_vector(self,sentence):
        # convert each character to vector of 1s and 0s
        # get character index from dictionary
        sentence_vector = []
        for char in sentence:
            char_vec = self.convert_char_to_vector(char)
            sentence_vector.append(char_vec)
        sentence_vector =  np.array(sentence_vector)
        # print("sentence_vector",sentence_vector.shape)
        return sentence_vector
    def convert_label_to_diacritic(self,indices):
        diacritic_vector = []
        for index in indices:
            label_vec = DataPreprocessing.INDEX2DIACRITIC[index]
            diacritic_vector.append(label_vec)
        diacritic_vector = np.array(diacritic_vector)
        return diacritic_vector
    
    def merge_sentence_diacritic(self,diacritic_vector,sentence):
        merged_sentence = ''
        i = 0
        while i < len(diacritic_vector):
            merged_sentence += sentence[i]+diacritic_vector[i]
            i+=1
        return merged_sentence
    def primary_diacritics_corrections(self,predicted_diacritized_string):
        ARABIC_LETTERS = frozenset([chr(x) for x in (list(range(0x0621, 0x63B)) + list(range(0x0641, 0x064B)))])
        ARABIC_LETTERS = ARABIC_LETTERS.union({' '})
        corrected_string = list(predicted_diacritized_string)
        i=0
        while i < len(corrected_string):
            if corrected_string[i] == ' ':
                while i+1 < len(corrected_string) and corrected_string[i+1] not in ARABIC_LETTERS:
                    corrected_string.pop(i+1)
                

            if corrected_string[i] == 'إ':
                while i+1 < len(corrected_string) and corrected_string[i+1] not in ARABIC_LETTERS:
                    corrected_string.pop(i+1)
                corrected_string.insert(i+1, 'ِ')
                # print(corrected_string)
                # print("here1")
            
            if corrected_string[i] in ['ى','ة'] and corrected_string[i-1] != 'َ':
                # print(corrected_string)
                while corrected_string[i-1] not in ARABIC_LETTERS and corrected_string[i-1] != 'ّ':
                    corrected_string.pop(i-1)
                    i-=1
                corrected_string.insert(i, 'َ')
                i+=1
                # print(corrected_string)
                # print("here2")

            if corrected_string[i] == 'ا' and not((i+1 < len(corrected_string) and corrected_string[i+1]==' ') or (i+2 < len(corrected_string) and corrected_string[i+1] == 'َ' and corrected_string[i+2]==' ') or (i+2 < len(corrected_string) and corrected_string[i+1] == 'ً' and corrected_string[i+2]==' ')):
                while i+1 < len(corrected_string) and corrected_string[i+1] not in ARABIC_LETTERS:
                    # print(corrected_string[i])
                    i+=1
                if i+1 < len(corrected_string) and corrected_string[i+1] == ' ':
                    while corrected_string[i] not in ARABIC_LETTERS: 
                        corrected_string.pop(i)
                        i-=1
                # print("here3")
                # print(corrected_string)

            if corrected_string[i] == 'ا' and corrected_string[i-1] != ' ' and i!=0:
                j = i
                while j+1 < len(corrected_string) and corrected_string[j+1] not in ARABIC_LETTERS:
                    j+=1
                if j+1 < len(corrected_string) and corrected_string[j+1] != ' ':
                    if corrected_string[i-1] not in ARABIC_LETTERS and corrected_string[i-1] != 'ّ': 
                        corrected_string.pop(i-1)
                        i-=1
                    corrected_string.insert(i, 'َ')
                    i+=1
                # print(corrected_string)
                # print("here4")
            elif i+1 < len(corrected_string) and corrected_string[i] in [' ','ى','آ','ا']:
                while i+1 < len(corrected_string) and corrected_string[i+1] not in ARABIC_LETTERS:
                    corrected_string.pop(i+1)
                # print(corrected_string)
                # print("here8")

                
            if i+1 < len(corrected_string) and corrected_string[i+1] == 'ْ' and (corrected_string[i-1] == ' ' or i==0):
                corrected_string.pop(i+1)
                # print(corrected_string)
                # print("here5")

            if (corrected_string[i] in ['ٍ','ٌ'] and i+1 < len(corrected_string) and corrected_string[i+1] != ' ') or (corrected_string[i]=='ً' and i+1 < len(corrected_string) and corrected_string[i+1] != 'ا'):
                corrected_string.pop(i)
                i-=1
                # print(corrected_string)
                # print("here6")

            if i+2 < len(corrected_string) and corrected_string[i] not in ['ء','ة','ا'] and corrected_string[i+1] == 'ً' and  corrected_string[i+2] == ' ':
                corrected_string.pop(i+1)
                # print(corrected_string)
                # print("here7")
            i+=1
        return ''.join(corrected_string)
        

    def Shadda_Corrections(self,predicted_diacritized_string):
        forbidden_char = [' ','ا','أ','آ','ى','ئ','ء','إ','ة']
        ARABIC_LETTERS = frozenset([chr(x) for x in (list(range(0x0621, 0x63B)) + list(range(0x0641, 0x064B)))])
        ARABIC_LETTERS = ARABIC_LETTERS.union({' '})
        corrected_string = list(predicted_diacritized_string)
        i=0
        while i < len(corrected_string):
            char = corrected_string[i]
            if char in forbidden_char or i==0 or corrected_string[i-1]==" ":
                if i + 1 < len(corrected_string) and corrected_string[i+1] == 'ّ':
                    corrected_string.pop(i+1)
                    if i + 2 < len(corrected_string) and corrected_string[i+2] not in ARABIC_LETTERS:
                        corrected_string.pop(i+1)

                elif i + 2 < len(corrected_string) and corrected_string[i+2] == 'ّ' and corrected_string[i+1] not in ARABIC_LETTERS:
                    corrected_string.pop(i+1)
                    corrected_string.pop(i+1)
            i += 1
        return ''.join(corrected_string)




# # define constants
# DIACRITIC_NAMES = ['Fathatan', 'Dammatan', 'Kasratan', 'Fatha', 'Damma', 'Kasra', 'Shadda', 'Sukun']
# NAME2DIACRITIC = dict((name, chr(code)) for name, code in zip(DIACRITIC_NAMES, range(0x064B, 0x0653)))
# # append shadda to diacritics
# NAME2DIACRITIC['Shadda_Dammatan'] = NAME2DIACRITIC['Shadda'] + NAME2DIACRITIC['Dammatan']
# NAME2DIACRITIC['Shadda_Kasratan'] = NAME2DIACRITIC['Shadda'] + NAME2DIACRITIC['Kasratan']
# NAME2DIACRITIC['Shadda_Fatha'] = NAME2DIACRITIC['Shadda'] + NAME2DIACRITIC['Fatha']
# NAME2DIACRITIC['Shadda_Damma'] = NAME2DIACRITIC['Shadda'] + NAME2DIACRITIC['Damma']
# NAME2DIACRITIC['Shadda_Kasra'] = NAME2DIACRITIC['Shadda'] + NAME2DIACRITIC['Kasra']
# NAME2DIACRITIC['Shadda_Fathatan'] = NAME2DIACRITIC['Shadda'] + NAME2DIACRITIC['Fathatan']

# DIACRITIC2NAME = dict((code, name) for name, code in NAME2DIACRITIC.items())
# ARABIC_DIACRITICS = frozenset(NAME2DIACRITIC.values())
# ARABIC_LETTERS = frozenset([chr(x) for x in (list(range(0x0621, 0x63B)) + list(range(0x0641, 0x064B)))])
# ARABIC_SYMBOLS = ARABIC_LETTERS | ARABIC_DIACRITICS
# SENTENCE_SEPARATORS = ';,،؛.:؟!'

# SENTENCE_TOKENIZATION_REGEXP = re.compile(r'([' + SENTENCE_SEPARATORS + r'])(?!\w)')
# # define SENTENCE_SEPARATORS as a set of characters
# SENTENCE_SEPARATORS = set(SENTENCE_SEPARATORS)
# ARABIC_SYMBOLS = ARABIC_SYMBOLS | SENTENCE_SEPARATORS
# # add space to arabic symbols
# ARABIC_SYMBOLS = ARABIC_SYMBOLS | {' '}
# SPACES = ' \t'
# EXTRA_SUKUN_REGEXP = re.compile(r'(?<=ال)' + NAME2DIACRITIC['Sukun'])

# # YA_REGEXP = re.compile(r'ى(?=['+''.join(ARABIC_DIACRITICS)+r'])')
# DIACRITIC_SHADDA_REGEXP = re.compile('(['+''.join(ARABIC_DIACRITICS)+'])('+NAME2DIACRITIC['Shadda']+')')


# def clean(text: str):
#     """
#     Clean text from non Arabic characters
#     :param text: input text
#     :return: cleaned text
#     """
#     assert isinstance(text, str)
#     line =  ''.join([c for c in text if c in ARABIC_SYMBOLS])
#     line = fix_errors(line)
#     # remove extra spaces
#     line = re.sub(r'\s+', ' ', line)
#     return line
# def fix_errors(diacritized_text):
#     assert isinstance(diacritized_text, str)
#     # Remove the extra Sukun from ال
#     diacritized_text = EXTRA_SUKUN_REGEXP.sub('', diacritized_text)
#     # Fix misplaced Fathatan
#     diacritized_text = diacritized_text.replace('اً', 'ًا')
#     # Fix reversed Shadda-Diacritic
#     diacritized_text = DIACRITIC_SHADDA_REGEXP.sub(r'\2\1', diacritized_text)
#     # Fix ى that should be ي (disabled)
#     # diacritized_text = YA_REGEXP.sub('ي', diacritized_text)
#     # Remove the duplicated diacritics by leaving the second one only when there are two incompatible diacritics
#     fixed_text = diacritized_text[0]
#     for x in diacritized_text[1:]:
#         if x in ARABIC_DIACRITICS and fixed_text[-1] in ARABIC_DIACRITICS:
#             if fixed_text[-1] != NAME2DIACRITIC['Shadda'] or x == NAME2DIACRITIC['Shadda']:
#                 fixed_text = fixed_text[:-1]
#         # Remove the diacritics that are without letters
#         elif x in ARABIC_DIACRITICS and fixed_text[-1] not in ARABIC_LETTERS:
#             continue
#         fixed_text += x
#     return fixed_text

# # load train and test data text
# def load_text():
#     # load train text
#     train_text = []
#     with open('dataset/train.txt','rt', encoding='utf-8') as f:
#         train_text_line = f.readline().strip()
#         # train_text.append(train_text_line)
#         while(train_text_line != ""):
#             train_text.append(train_text_line)
#             train_text_line = f.readline().strip()

#     # load test text
#     test_text = []
#     with open('dataset/val.txt', encoding='utf-8') as f:
#         test_text_line = f.readline().strip()
#         while test_text_line != "":
#             test_text.append(test_text_line)
#             test_text_line = f.readline().strip()
#     return train_text, test_text
# def clear_diacritics(text: str):
#     assert isinstance(text, str)
#     return ''.join([l for l in text if l not in ARABIC_DIACRITICS])

# def text_preprocessing(text,name,debug = False):
#     # tokenize text to sentences
#     sentences = []
#     # loop on each line "document"
#     for i in range(len(text)):
#         # get line
#         line = text[i]

#         # clean line from non arabic characters
#         line = clean(line)

#         # tokenize line to sentences using tokenizer or regex
#         line_sentences = sentence_tokenizer(line)

#         if len(line_sentences) > 1:
#             for i in range(0,len(line_sentences),2):
#                 if i+1 < len(line_sentences):
#                     sentences.append(line_sentences[i]+line_sentences[i+1])
#                 else:
#                     sentences.append(line_sentences[i])
#         else:
#             sentences.extend(line_sentences)

#     filtered_sentences = sentences
#     # loop on each sentence

#     # for i in range(len(sentences)):
#     #     # filter sentence
#     #     possible_sentences = filter_tokenized_sentence(tokenize(fix_diacritics_errors(sentences[i])))
#     #     if len(possible_sentences) > 0:
#     #         filtered_sentences.append(' '.join(possible_sentences))

#     # save filtered sentences in text file
#     with open("dataset/"+name, 'w', encoding='utf-8') as f:
#         for sentence in filtered_sentences:
#             f.write(remove_punctuation(sentence).strip()+'\n')
#     with open("dataset/undiacritized_"+name, 'w', encoding='utf-8') as f:
#         for sentence in filtered_sentences:
#             f.write(clear_diacritics( remove_punctuation(sentence)).strip()+'\n')

#     return filtered_sentences

# def sentence_tokenizer(text,debug = False):
#     # first split document to sentences on end of sentence characters on
#     sentences_splits =  re.split(r'\n', text)
    
#     # loop on each sentence
#     sentences = []
#     for sentence in sentences_splits:
#         if sentence is not None:
#             # check if sentence is larger than 600 characters
#             if len(sentence) > 600:
#                 # split sentence on ;,،؛.:؟!
#                 sentences_splits2 = re.split(SENTENCE_TOKENIZATION_REGEXP, sentence)
#                 for sentence2 in sentences_splits2:
#                     if sentence2 is not None:
#                         if sentence2.strip(SPACES) != '':
#                             if len(sentence2) > 20:
#                                 sentences.append(sentence2)
#             else:
#                 if sentence.strip(SPACES) != '':
#                     if len(sentence) > 20:
#                         sentences.append(sentence)
#     return sentences

# def remove_punctuation(text):
#     assert isinstance(text, str)
#     line =  ''.join([l for l in text if l not in SENTENCE_SEPARATORS])
#     line = re.sub(r'\s+', ' ', line)
#     return line
# def remove_non_arabic_chars(text):
#     assert isinstance(text, str)
#     line =  ''.join([c for c in text if c in ARABIC_LETTERS|{' '}])

#     # remove extra spaces
#     line = re.sub(r'\s+', ' ', line)
#     return line


# def extract_diacritics_with_previous_letter(text):
#     assert isinstance(text, str)
#     diacritics_list = []
#     i = 0
#     sentence = ''
#     while i <len(text):
#         # check if the character is a arabic letter
#         if text[i] in ARABIC_LETTERS:
#             # check if next character is diacritic not shadda
#             # add arabic letter to sentence
#             sentence += text[i]
#             if i+1 < len(text):
#                 if text[i+1] in ARABIC_DIACRITICS - {NAME2DIACRITIC['Shadda']}:
#                     diacritics_list.append([text[i], text[i+1]])
#                     i += 1
#                 elif text[i+1] == NAME2DIACRITIC['Shadda'] and i+2< len(text) and \
#                       text[i+2] in ARABIC_DIACRITICS - {NAME2DIACRITIC['Shadda']} :
#                     diacritics_list.append([text[i], text[i+1]+text[i+2]])
#                     i += 2
#                 elif text[i+1] == NAME2DIACRITIC['Shadda']:
#                     diacritics_list.append([text[i], text[i+1]])
#                     i += 1
#                 else:
#                     diacritics_list.append([text[i], ''])
#                 i+=1    
#             else:
#                 diacritics_list.append([text[i], ''])
#                 i+=1
#         elif text[i] == ' ':
#             sentence += text[i]
#             diacritics_list.append([' ', ''])
#             i+=1
#         else:
#             i+=1
#     return diacritics_list,sentence
# def test(labels,reference,undiacritized_sentence):
#     sentence =''
#     for i in range(len(labels)):
#         sentence += undiacritized_sentence[i]+labels[i][1]
#     print("sentence",sentence)
#     print("reference",reference)
    
#     assert sentence == reference

# def load_text(file_path):
#     # read text file sentence by sentence
#     sentences = []
#     with open(file_path,'rt', encoding='utf-8') as f:
#         text_line = f.readline().strip()
#         # train_text.append(train_text_line)
#         while(text_line != ""):
#             sentences.append(text_line)
#             text_line = f.readline().strip()
#     return sentences


             
# # train,test = load_text()
# # filtered_train = text_preprocessing(train,"train_preprocessed.txt")


# data = load_text("dataset/train_preprocessed.txt")
# labels,sentence= extract_diacritics_with_previous_letter(data[0])
# print(sentence)
# test(labels,data[0],sentence)
# # # print 10 lines of train text
# # for i in range(10):
# #     print(filtered_train[i])

# # cleaned_train = clean(train)

# # print 10 lines of train text
# # for i in range(10):
# #     print(clean(train[i]))


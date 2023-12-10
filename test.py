import textProcessing as tp
import nltk
from tokenizer import Tokenizer
from utilities import load_text, extract_labels, create_tokenized_sentence, create_tokenized_sentence2


# train_text = load_text("dataset/train.txt")
# tp.preprocessing_text(train_text,"train_preprocessed.txt")

# extract diacritics from train text
train_text = load_text("dataset/train_preprocessed.txt")
train_labels = extract_labels(train_text)

# train_labels = tp.extract_diacritics_with_previous_letter(train_text)

# train_text, test_text = tp.loadText()
# filtered_sentences = tp.preprocessing_text(train_text,"train_preprocessed.txt")

sentences = []
with open('dataset/undiacritized_train_preprocessed.txt','rt', encoding='utf-8') as f:
    text_line = f.readline().strip()
    # train_text.append(train_text_line)
    while(text_line != ""):
        sentences.append(text_line)
        text_line = f.readline().strip()

# load undiacritized train text

# create tokenizer
tokenizer = Tokenizer("pyarabic")
tokenized_sentences = tokenizer.tokenize_sentences(sentences)
# print first 10 sentences row by row
# for i in range(10):
#     tp.printFirstLine(tokenized_sentences[i])


# merge tokens with diacritics
merged_sentences = []
for i in range(len(tokenized_sentences)):
    merged_sentences.append(tp.merge_tokenized_with_diacritics3(tokenized_sentences[i],train_labels[i]))
for i in range(10):
    tp.printFirstLine(merged_sentences[i])

# compare with original sentences print error if different
count_errors = 0
for i in range(len(train_text)):
    # remove more than one space in train text
    train_text[i] = " ".join(train_text[i].split())
    if train_text[i].strip() != merged_sentences[i].strip():
        if count_errors < 10:
            print("Error: sentence not equal to merged sentence")
            print("Original sentence: ")
            tp.printFirstLine(train_text[i])
            print("Merged sentence: ")
            tp.printFirstLine(merged_sentences[i])
        # print("Error: sentence not equal to merged sentence")
        # print("Original sentence: ",train_text[i])
        # print("Merged sentence: ",merged_sentences[i])
        count_errors += 1

print("Number of errors: ",count_errors)

        


# print(len(train_text))

# line = train_text[1]
# print("line: ",line)
# line = tp.clean_text(line)
# print("cleaned line: ",line)
# line_sentences = tp.sentence_tokenizer(line,debug=True)

# sentences = []
# if len(line_sentences) > 1:
#     for i in range(0,len(line_sentences),2):
#         if i+1 < len(line_sentences):
#             sentences.append(line_sentences[i]+line_sentences[i+1])
#         else:
#             sentences.append(line_sentences[i])
# else:
#     sentences.extend(line_sentences)

# for t in sentences:
#     print("t: ", t)

# tp.preprocessing_text(test_text,"test_preprocessed.txt")
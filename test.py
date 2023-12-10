import textProcessing as tp
import nltk
from tokenizer import Tokenizer

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
print("tokenized_sentences: ",tokenized_sentences[:10])


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
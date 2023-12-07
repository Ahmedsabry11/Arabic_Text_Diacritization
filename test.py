import textProcessing as tp
import nltk


train_text, test_text = tp.loadText()
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

tp.preprocessing_text(train_text,"train_preprocessed.txt")
tp.preprocessing_text(test_text,"test_preprocessed.txt")
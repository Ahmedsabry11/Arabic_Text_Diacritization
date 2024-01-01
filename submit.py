from cbhg_Training import CBHGTrainer
from data_preprocessing import DataPreprocessing
import pickle
from output_file import OutputFile
from LSTM_Training import LSTMTrainer
if __name__ == "__main__":
    lstmTrainer = LSTMTrainer(epoch=9,load=True)
    # cbhgTrainer.calcluate_accuracy_nopadding(with_correction=True)
    # print(cbhgTrainer.predict("السلام عليكم ورحمة الله وبركاته"))
    dataPreprocessor = DataPreprocessing()
    csv_writer = OutputFile(test_set_without_labels_file="dataset/test_set_without_labels.csv",test_set_with_labels_file="dataset/test_set_gold.csv")
        
    count = 0
    test_data = dataPreprocessor.load_text("dataset/test_no_diacritics.txt") 
    sentences = dataPreprocessor.predict_sentence_tokenizer(test_data)
    # write to file

    # with open("dataset/test_preprocessed3.txt","w",encoding="utf-8") as file:
    #     for i in range(len(sentences)):
    #         line_sentences = sentences[i]
    #         final_sentence = ""
    #         for sentence in line_sentences:
    #             file.write(sentence+"\n")

    
    for i in range(len(sentences)):
        line_sentences = sentences[i]
        final_sentence = ""
        for sentence in line_sentences:
            final_sentence += lstmTrainer.predict(sentence)
            
        csv_writer.char_with_diacritic_csv(final_sentence)
        with open("dataset/test_with_diacritics.txt","a",encoding="utf-8") as file:
            file.write(final_sentence+"\n")
        if count % 100 == 0:
            print("processed ",(count/2500.0) * 100)
        count += 1

    # with open("dataset/test_no_diacritics.txt","rt",encoding="utf-8") as file:
    #     for line in file:
    #         sentence = cbhgTrainer.predict(line.strip())
    #         # write to file
    #         with open("dataset/test_with_diacritics.txt","a",encoding="utf-8") as file:
    #             file.write(sentence+"\n")
    #         if count % 100 == 0:
    #             print("processed ",(count/2500.0) * 100)
    #         count += 1

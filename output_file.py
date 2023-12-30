import pickle
import csv
from data_preprocessing import DataPreprocessing

class OutputFile():
    def __init__(self) -> None:
        self.line_num = 0
        self.diactritic2index = pickle.load(open('diacritic2id.pickle', 'rb'))
        self.test_set_without_labels_file = "sample_test_set_without_labels.csv"
        self.test_set_with_labels_file = "test_set_with_golds.csv"
        self.dataPreprocessor = DataPreprocessing()
        with open(self.test_set_with_labels_file, "w") as fout:
                # if you need to add any head to the file
            pass

    def create_mapping_by_line_number(self,line_number):
        id_letter_mapping = []

        with open(self.test_set_without_labels_file, 'r',encoding='utf-8') as file:
            reader = csv.DictReader(file)
            # Process rows where the 'line_number' column matches the specified value
            for row in reader:
                if row['line_number'] == str(line_number):
                    id_letter_mapping.append((row['letter'], row['id']))
        return id_letter_mapping
    
    def char_with_diacritic_csv(self,sentence):
        fieldnames = ['ID', 'label']
        char2id = self.create_mapping_by_line_number(self.line_num)
        # print(len(char2id))
        labels,_= self.dataPreprocessor.extract_diacritics_with_previous_letter('s'+sentence,False)
        with open(self.test_set_with_labels_file,"a",newline='') as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            if file.tell() == 0:
                writer.writeheader()
            for i,(char,id) in enumerate(char2id):
                label_id = self.diactritic2index[labels[i]]
                new_row_data = {'ID': id, 'label': label_id}
                writer.writerow(new_row_data)
        self.line_num += 1



if __name__ == "__main__":
    predicted_string = "قُوَّةُ الإِرَادَةِ"
    output_file = OutputFile()
    output_file.char_with_diacritic_csv(predicted_string)
    output_file.char_with_diacritic_csv(predicted_string)

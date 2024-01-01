import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Dataset, DataLoader
from cbhg import CBHGModel
from data_preprocessing import DataPreprocessing
from cbhg_data_loader import MyDataset
from output_file import OutputFile


class CBHGTrainer:
    def __init__(self,load=False,epoch = 0,input_size = 39,hidden_size = 128,output_size = 16,batch_size = 32,num_epochs = 20):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.batch_size = batch_size
        self.num_epochs = num_epochs

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = CBHGModel( 
            inp_vocab_size = 39,
            targ_vocab_size = 16,
            embedding_dim = 512,
            use_prenet = True,
            prenet_sizes = [512, 256],
            cbhg_gru_units = 512,
            cbhg_filters = 16,
            cbhg_projections = [128, 256],
            post_cbhg_layers_units = [256, 256],
            post_cbhg_use_batch_norm = True
        )
        self.current_epoch = 0
        self.current_epoch = epoch
        if load:
            self.load_model(epoch)
        self.model.to(self.device)
        self.dataset = MyDataset(T = 280)
        # self.test_dataset = MyDataset(dataset_path="dataset/test_preprocessed.txt",T = 600)
        self.test_dataset = MyDataset(dataset_path="dataset/test_preprocessed2.txt",T = 300)
        self.train_dataloader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)
        self.test_dataloader = DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=True)
        self.csv_writer = OutputFile(test_set_without_labels_file="dataset/test_set_without_labels.csv",test_set_with_labels_file="dataset/test_set_gold.csv")
        self.criterion = nn.CrossEntropyLoss(ignore_index=15)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=0.001)
        self.scheduler = StepLR(self.optimizer, step_size=2, gamma=0.5)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=0.001)
        self.scheduler = StepLR(self.optimizer, step_size=2, gamma=0.5)
        

    def train(self):
        for epoch in range(0,self.current_epoch):
            self.scheduler.step()
        for epoch in range(self.current_epoch,self.num_epochs):
            self.model.train()
            running_loss = 0.0
            for i, data in enumerate(self.train_dataloader, 0):
                # get the inputs
                inputs, labels,_ = data
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                # zero the parameter gradients
                self.optimizer.zero_grad()
                # forward + backward + optimize
                outputs = self.model(inputs)
                outputs = outputs["diacritics"]
                # print("outputs size: ",outputs.size())
                outputs = outputs.view(-1, outputs.shape[-1])
                labels = labels.view(-1)
                loss = self.criterion(outputs, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
                self.optimizer.step()
                # print statistics
                running_loss += loss.item()
                if i % 2 == 0:  # print every 100 mini-batches
                    print('[%d, %5d] loss: %.3f' %
                          (epoch + 1, i + 1, running_loss / 2))
                    running_loss = 0.0
            self.scheduler.step()
            self.scheduler.step()
            # save the model if it has the best training loss till now
            self.save_model(epoch+1)
        print('Finished Training')

    def test(self):
        self.model.eval()
        with torch.no_grad():
            running_loss = 0.0
            for i, data in enumerate(self.test_dataloader, 0):
                # get the inputs
                inputs, labels,_ = data
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                # forward + backward + optimize
                outputs = self.model(inputs)
                outputs = outputs["diacritics"]
                outputs = outputs.view(-1, outputs.shape[-1])
                labels = labels.view(-1)
                loss = self.criterion(outputs, labels)
                running_loss += loss.item()
            print('Test loss: %.3f' %
                  (running_loss / len(self.test_dataloader)))
    def calcluate_accuracy(self):
        self.model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            for i, data in enumerate(self.test_dataloader, 0):
                # get the inputs
                inputs, labels,_ = data
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                # forward + backward + optimize
                outputs = self.model(inputs)
                outputs = outputs["diacritics"]
                outputs = outputs.view(-1, outputs.shape[-1])
                labels = labels.view(-1)
                _, predicted = torch.max(outputs.data, 1)
                # cut the padding
                predicted = predicted[labels != 15]
                labels = labels[labels != 15]
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            print('Accuracy of the network on the test set: %d %%' % (
                    100 * correct / total))
            print('Accuracy of the network on the test set: %f %%' % (
                    100 * correct / total))
        with torch.no_grad():
            correct = 0
            total = 0
            for i, data in enumerate(self.train_dataloader, 0):
                # get the inputs
                inputs, labels,_ = data
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                # forward + backward + optimize
                outputs = self.model(inputs)
                outputs = outputs["diacritics"]
                outputs = outputs.view(-1, outputs.shape[-1])
                labels = labels.view(-1)
                _, predicted = torch.max(outputs.data, 1)
                      # cut the padding 
                predicted = predicted[labels != 15]
                labels = labels[labels != 15]
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            print('Accuracy of the network on the train set: %d %%' % (
                    100 * correct / total))
            # print floating point accuracy
            print('Accuracy of the network on the train set: %f %%' % (
                    100 * correct / total))
    def calcluate_accuracy_nopadding(self,with_correction=False):
        self.model.eval()
        dataPreprocessor = DataPreprocessing()
        total_size = 0
        total_size = 0
        with torch.no_grad():
            correct = 0
            total = 0
            correct_after = 0
            total_after = 0
            for i, data in enumerate(self.test_dataloader, 0):
                # get the inputs
                inputs, labels,sentences = data
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                # forward + backward + optimize
                outputs = self.model(inputs)
                outputs = outputs["diacritics"]
                _, predicted = torch.max(outputs.data, 2)
                # cut the padding 
                # loop over the batch
                for j in range(len(predicted)):
                    correct_batch,index = self.get_correct(labels[j],predicted[j])
                    correct += correct_batch
                    total += index
                    if with_correction:
                        index = self.get_padding_index(labels[j])
                        if index == -1:
                            index = len(labels[j])
                        if index == -1:
                            index = len(labels[j])
                        prediction = predicted[j][:index]
                        label = labels[j][:index]
                        # convert to cpu
                        prediction = prediction.cpu()
                        label = label.cpu()
                        # convert to numpy
                        prediction = prediction.numpy()
                        label = label.numpy()
                        prediction = dataPreprocessor.convert_label_to_diacritic(prediction)
                        
                        # merge the sentence
                        sentence = sentences[j]
                        diacritized_sentence = dataPreprocessor.merge_sentence_diacritic(diacritic_vector= prediction,sentence=sentence)

                        # apply correction
                        corrected_sentence = dataPreprocessor.Shadda_Corrections(diacritized_sentence)
                        # corrected_sentence = dataPreprocessor.primary_diacritics_corrections(corrected_sentence)

                        # extract the label
                        corrected_label,sentence= dataPreprocessor.extract_diacritics_with_previous_letter(corrected_sentence)
                        corrected_label = dataPreprocessor.convert_labels_to_indices(corrected_label)
                        corrected_label = corrected_label.reshape(-1)
                        label = label.reshape(-1)
                        correct_after += np.sum(corrected_label == label)
                        total_after += len(corrected_label)
                        
                        
            print('Accuracy of the network on the test set: %d %%' % (
                    100 * correct / total))
            # print floating point accuracy
            print('Accuracy of the network on the test set: %f %%' % (
                    100 * correct / total))
            if with_correction:
                print('Accuracy of the network on the test set after correction: %d %%' % (
                        100 * correct_after / total_after))
                # print floating point accuracy
                print('Accuracy of the network on the test set after correction: %f %%' % (
                        100 * correct_after / total_after))
                print("total: ",total)
                print("total_after: ",total_after)
                print("correct: ",correct)
                print("correct_after: ",correct_after)
                assert total == total_after
                print("total: ",total)
                print("total_after: ",total_after)
                print("correct: ",correct)
                print("correct_after: ",correct_after)
                assert total == total_after
    def get_correct(self,label,prediction):
        # find index of torch label that has value = 15
        label = label.view(-1)
        index = torch.where(label == 15)[0]
        # check if there is no padding
        if index.size() == torch.Size([0]):
            return (prediction == label).sum().item(),label.size(0)
        # get first index
        index = index[0]
        # cut the padding
        prediction = prediction[:index]
        label = label[:index]


        # calculate correct
        correct = (prediction == label).sum().item()
        return correct,label.size(0)
        return correct,label.size(0)
    def get_padding_index(self,label):
        # find index of torch label that has value = 15
        label = label.view(-1)
        index = torch.where(label == 15)[0]
        # check if there is no padding
        if index.size() == torch.Size([0]):
            return -1
        # get first index
        index = index[0]
        return index
    def predict(self,sentence):
        dataPreprocessor = DataPreprocessing()
        # should clean the sentence
        sentence = dataPreprocessor.remove_non_arabic_chars(sentence)
        final_sentence = ""
        self.model.eval()
        with torch.no_grad():
            inputs = dataPreprocessor.convert_sentence_to_indices(sentence)
            inputs = torch.LongTensor(inputs)
            inputs = inputs.unsqueeze(0)
            inputs = inputs.to(self.device)
            outputs = self.model(inputs)
            outputs = outputs["diacritics"]
            outputs = outputs.view(-1, outputs.shape[-1])
            _, predicted = torch.max(outputs.data, 1)

            # convert to cpu
            predicted = predicted.cpu()
            # convert to numpy
            predicted = predicted.numpy()

            predicted = predicted.reshape(-1)

            # convert to diacritics
            predicted = dataPreprocessor.convert_label_to_diacritic(predicted)

            # merge the sentence
            diacritized_sentence = dataPreprocessor.merge_sentence_diacritic(diacritic_vector= predicted,sentence=sentence)

            # apply correction
            corrected_sentence = dataPreprocessor.Shadda_Corrections(diacritized_sentence)

            # call csv writer
            # self.csv_writer.char_with_diacritic_csv(corrected_sentence)

            final_sentence = corrected_sentence
            
    
        return final_sentence

        
    def save_model(self,epoch):
        torch.save(self.model.state_dict(), "models/cbhg_model_"+str(epoch)+".pth")
    def load_model(self,epoch=9):
        self.model.load_state_dict(torch.load("models/cbhg_model_"+str(epoch)+".pth"))
        self.model.eval()
        self.current_epoch = epoch



# if __name__ == "__main__":
#     cbhgTrainer = CBHGTrainer(epoch=7,load=True)
#     # cbhgTrainer.train()
#     # cbhgTrainer.test()
#     # cbhgTrainer.calcluate_accuracy()
#     cbhgTrainer.calcluate_accuracy_nopadding(with_correction=True)
#     # print(cbhgTrainer.predict("السلام عليكم ورحمة الله وبركاته"))

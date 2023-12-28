import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Dataset, DataLoader
from LSTM import LSTMClassifier
from data_preprocessing import DataPreprocessing
from DatasetLoader import MyDataset

class LSTMTrainer:
    def __init__(self,load=True,epoch = 0,input_size = 39,hidden_size = 128,output_size = 16,batch_size = 512,num_epochs = 20):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.batch_size = batch_size
        self.num_epochs = num_epochs

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.model = LSTMClassifier(self.input_size, self.hidden_size, self.output_size)
        self.current_epoch = 0
        self.current_epoch = epoch
        if load:
            self.load_model(epoch)
        self.model.to(self.device)
        self.dataset = MyDataset(T = 280)
        self.test_dataset = MyDataset(dataset_path="dataset/test_preprocessed.txt",T = 280)
        self.train_dataloader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)
        self.test_dataloader = DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=True)
        self.criterion = nn.CrossEntropyLoss(ignore_index=15)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.scheduler = StepLR(self.optimizer, step_size=2, gamma=0.5)
        

    def train(self):
        for epoch in range(0,self.current_epoch):
            self.scheduler.step()
        for epoch in range(self.current_epoch,self.num_epochs):
            self.model.train()
            running_loss = 0.0
            for i, data in enumerate(self.train_dataloader, 0):
                # get the inputs
                inputs, labels = data
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                # zero the parameter gradients
                self.optimizer.zero_grad()
                # forward + backward + optimize
                outputs = self.model(inputs)
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
            # save the model if it has the best training loss till now
            self.save_model(epoch+1)
        print('Finished Training')

    def test(self):
        self.model.eval()
        with torch.no_grad():
            running_loss = 0.0
            for i, data in enumerate(self.test_dataloader, 0):
                # get the inputs
                inputs, labels = data
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                # forward + backward + optimize
                outputs = self.model(inputs)
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
                inputs, labels = data
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                # forward + backward + optimize
                outputs = self.model(inputs)
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
                inputs, labels = data
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                # forward + backward + optimize
                outputs = self.model(inputs)
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
    def save_model(self,epoch):
        torch.save(self.model.state_dict(), "models/lstm_model_"+str(epoch)+".pth")
    def load_model(self,epoch=9):
        self.model.load_state_dict(torch.load("models/lstm_model_"+str(epoch)+".pth"))
        self.model.eval()
        self.current_epoch = epoch



if __name__ == "__main__":
    lstmTrainer = LSTMTrainer()
    lstmTrainer.train()
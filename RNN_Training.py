import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Dataset, DataLoader
from RNN import RNNClassifier
from data_preprocessing import DataPreprocessing
from DatasetLoader import MyDataset

class RNNTrainer:
    def __init__(self,input_size = 39,hidden_size = 128,output_size = 16,batch_size = 512,num_epochs = 10):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.batch_size = batch_size
        self.num_epochs = num_epochs

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.model = RNNClassifier(self.input_size, self.hidden_size, self.output_size)
        self.model.to(self.device)
        self.dataset = MyDataset()
        self.train_dataloader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)
        self.criterion = nn.CrossEntropyLoss(ignore_index=15)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.scheduler = StepLR(self.optimizer, step_size=5, gamma=0.5)

    def train(self):
        for epoch in range(self.num_epochs):
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
                if i % 100 == 99:  # print every 100 mini-batches
                    print('[%d, %5d] loss: %.3f' %
                          (epoch + 1, i + 1, running_loss / 100))
                    running_loss = 0.0
            self.scheduler.step()
            # save the model if it has the best training loss till now
            self.save_model()
        print('Finished Training')
    def save_model(self):
        torch.save(self.model.state_dict(), "models/rnn_model.pth")
            



if __name__ == "__main__":
    rnnTrainer = RNNTrainer()
    rnnTrainer.train()
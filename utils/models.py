import torch.nn as nn
import torch.nn.functional as F


# define the NN architecture
class ConvDenoiser(nn.Module):
    def __init__(self, num_classes=4):
        super(ConvDenoiser, self).__init__()
        ## encoder layers ##

        self.conv1 = nn.Conv1d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(32)
        self.pool1 = nn.MaxPool1d(2)
         # Lout = [(Lin + 2 * padding - kernel_size)/stride + 1]  --> Lout = [(32 + 2*1 - 3)/1 + 1 --> 32
         # pooling -> 16

        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(64)
        self.pool2 = nn.MaxPool1d(2)
        # Lout = (16 + 2 - 3)/1 + 1 --> Lout = 16
        # pooling --> 8

        ## decoder layers ##
        # Lout = (Lin - 1) * stride - 2 * padding + kernel_size + out_padding
        self.t_conv1 = nn.ConvTranspose1d(64, 128, kernel_size=2, stride=2)
        # Lout = (8 - 1) * 2 - 2 * 0 + 2 + 0 --> Lout= 16
        self.t_conv2 = nn.ConvTranspose1d(128, 64, kernel_size=2, stride=2)  # Double the length
        # Lout = (16 - 1) * 2 - 2 * 0 + 2 + 0 --> Lout = 32

        self.conv_out = nn.Conv1d(64, 1, kernel_size=3, padding=1)


    def forward(self, x):
        ## encode ##
        # add hidden layers with relu activation function
        # and maxpooling after
        z = self.pool1(F.relu(self.bn1(self.conv1(x))))
        z = self.pool2(F.relu(self.bn2(self.conv2(z))))

        # 27/01/2025:  TODO line 42 --> instead of X --> Z in order for the classifier to take this variable
        # todo --> std
        # todo 4 different encoder decoders and combination

        ## decode ##
        # add transpose conv layers, with relu activation function
        x = F.relu(self.t_conv1(z))
        x = F.relu(self.t_conv2(x))
        # transpose again, output should have a sigmoid applied
        x = F.sigmoid(self.conv_out(x))


        return x, z


class Classifier(nn.Module):
    def __init__(self, num_classes=4):
        super(Classifier, self).__init__()
        self.conv1 = nn.Conv1d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(32)
        self.pool1 = nn.MaxPool1d(2)
         # Lout = [(Lin + 2 * padding - kernel_size)/stride + 1]  --> Lout = [(32 + 2*1 - 3)/1 + 1 --> 32
         # pooling -> 16

        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(64)
        self.pool2 = nn.MaxPool1d(2)
        # Lout = 16
        # pooling --> 8

        self.conv3 = nn.Conv1d(64, 32, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(32)
        self.pool3 = nn.MaxPool1d(2)
        # Lout = 8
        # pooling --> 4

        # Fully connected layer for classification
        self.fc1 = nn.Linear(32*4, num_classes) # input size --> 32
        self.softmax = nn.Softmax(dim=1)

    def forward(self, z):
        z = self.pool1(F.relu(self.bn1(self.conv1(z))))
        z = self.pool2(F.relu(self.bn2(self.conv2(z))))
        z = self.pool3(F.relu(self.bn3(self.conv3(z))))

        z = z.view(z.size(0), -1)  # Flatten for fully connected layer
        z = self.fc1(z)
        z = self.softmax(z)  # Apply softmax to get class probabilities

        return z


    # def forward(self, x):
    #     x = self.pool1(F.relu(self.bn1(self.conv1(x))))
    #     x = self.pool2(F.relu(self.bn2(self.conv2(x))))
    #     x = self.pool3(F.relu(self.bn3(self.conv3(x))))
    #
    #     x = x.view(x.size(0), -1)  # Flatten for fully connected layer
    #     x = self.fc1(x)
    #     x = self.softmax(x)  # Apply softmax to get class probabilities
    #
    #     return x
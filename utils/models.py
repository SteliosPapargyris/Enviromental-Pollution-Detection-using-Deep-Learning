import torch.nn as nn
import torch.nn.functional as F


class ConvDenoiser(nn.Module):
    def __init__(self):
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

        # input length --> 32
        # After pool1: 16, After pool2: 8, 64 channels
        self.flattened_size = 64 * 8  # update based on input size
        self.fc = nn.Linear(self.flattened_size, self.flattened_size)  # you can change output size too

        ## decoder layers ##
        # Lout = (Lin - 1) * stride - 2 * padding + kernel_size + out_padding
        self.t_conv1 = nn.ConvTranspose1d(64, 64, kernel_size=2, stride=2)
        # Lout = (8 - 1) * 2 - 2 * 0 + 2 + 0 --> Lout= 16
        self.t_conv2 = nn.ConvTranspose1d(64, 32, kernel_size=2, stride=2, output_padding=1)  # Double the length
        # Lout = (16 - 1) * 2 - 2 * 0 + 2 + 0 --> Lout = 32

        self.conv_out = nn.Conv1d(32, 1, kernel_size=3, padding=1)


    def forward(self, x):
        ## encode ##
        # add hidden layers with relu activation function
        # and maxpooling after
        z = self.pool1(F.leaky_relu(self.bn1(self.conv1(x))))
        z = self.pool2(F.leaky_relu(self.bn2(self.conv2(z))))
        z_latent = z.clone()

 # Flatten and pass through dense layer
        batch_size = z.size(0)
        z_flat = z.view(batch_size, -1)
        z_dense = F.leaky_relu(self.fc(z_flat))

        # Unflatten before decoding
        z_unflat = z_dense.view(batch_size, 64, 8)

        ## decode ##
        # add transpose conv layers, with leaky relu activation function
        x = F.leaky_relu(self.t_conv1(z_unflat))
        x = F.leaky_relu(self.t_conv2(x))
        # transpose again, output should have a sigmoid applied
        x = F.sigmoid(self.conv_out(x))

        return x, z_latent

class LinearDenoiser(nn.Module):
    def __init__(self, input_size=33):
        super(LinearDenoiser, self).__init__()
        
        self.input_size = input_size
        
        ## encoder layers ##
        # First layer: input_size -> smaller representation
        self.linear1 = nn.Linear(33, 32)
        self.bn1 = nn.BatchNorm1d(32)
        self.dropout1 = nn.Dropout(0.2)
        
        # Second layer: further compression
        self.linear2 = nn.Linear(32, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.dropout2 = nn.Dropout(0.2)
        
        # Bottleneck layer: most compressed representation
        self.linear3 = nn.Linear(64, 32)
        self.bn3 = nn.BatchNorm1d(32)
        
        ## decoder layers ##
        # Expand back to original dimensions
        self.linear4 = nn.Linear(32, 64)
        self.bn4 = nn.BatchNorm1d(64)
        self.dropout3 = nn.Dropout(0.2)
        
        self.linear5 = nn.Linear(64, 128)
        self.bn5 = nn.BatchNorm1d(128)
        self.dropout4 = nn.Dropout(0.2)
        
        # Output layer: back to original input size
        self.linear_out = nn.Linear(128, input_size)
        
    def forward(self, x):
        # x shape: (batch_size, 1, sequence_length) for 1D conv input
        # We need to reshape for linear layers
        
        # Remove channel dimension and flatten if needed
        if len(x.shape) == 3:  # (batch, channels, length)
            x = x.squeeze(1)  # Remove channel dimension -> (batch, length)
        
        ## encoder ##
        x = F.leaky_relu(self.bn1(self.linear1(x)))
        x = self.dropout1(x)
        
        x = F.leaky_relu(self.bn2(self.linear2(x)))
        x = self.dropout2(x)
        
        z_latent = x.clone()

        # Bottleneck
        encoded = F.leaky_relu(self.bn3(self.linear3(x)))
        
        ## decoder ##
        x = F.leaky_relu(self.bn4(self.linear4(encoded)))
        x = self.dropout3(x)
        
        x = F.leaky_relu(self.bn5(self.linear5(x)))
        x = self.dropout4(x)
        
        # Output layer (no activation for reconstruction)
        x = self.linear_out(x)
        
        # Add channel dimension back if needed to match original input format
        x = x.unsqueeze(1)  # (batch, length) -> (batch, 1, length)
        
        return x, z_latent
    
class Classifier(nn.Module):
    def __init__(self, num_classes=4):
        super(Classifier, self).__init__()
        self.conv1 = nn.Conv1d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(32)
        self.pool1 = nn.MaxPool1d(2)
         # Lout = [(Lin + 2 * padding - kernel_size)/stride + 1]  --> Lout = [(128 + 2*1 - 3)/1 + 1 --> 128
         # pooling -> 64

        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(64)
        self.pool2 = nn.MaxPool1d(2)
        # Lout = 64
        # pooling --> 32

        self.conv3 = nn.Conv1d(64, 32, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(32)
        self.pool3 = nn.MaxPool1d(2)
        # Lout = 32
        # pooling --> 16

        # Fully connected layer for classification
        self.fc1 = nn.Linear(32*4, num_classes) # input size --> 32
        self.softmax = nn.Softmax(dim=1)

    def forward(self, z):
        z = self.pool1(F.leaky_relu(self.bn1(self.conv1(z))))
        z = self.pool2(F.leaky_relu(self.bn2(self.conv2(z))))
        z = self.pool3(F.leaky_relu(self.bn3(self.conv3(z))))

        z = z.view(z.size(0), -1)  # Flatten for fully connected layer
        z = self.fc1(z)
        z = self.softmax(z)  # Apply softmax to get class probabilities

        return z
import torch.nn as nn
import torch.nn.functional as F


class ConvDenoiser(nn.Module):
    def __init__(self, input_size=33, output_size=32):
        super(ConvDenoiser, self).__init__()

        self.input_size = input_size
        self.output_size = output_size
        
        ## encoder layers ##
        self.conv1 = nn.Conv1d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(32)
        self.pool1 = nn.AdaptiveMaxPool1d(input_size // 2)  # 33 -> 16
        
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(64)
        self.pool2 = nn.AdaptiveMaxPool1d(input_size // 4)  # 16 -> 8
        
        self.dropout = nn.Dropout(0.2)
        
        # More precise calculation
        self.encoded_length = input_size // 4
        self.flattened_size = 64 * self.encoded_length
        self.fc = nn.Linear(self.flattened_size, self.flattened_size // 2)
        self.fc_decode = nn.Linear(self.flattened_size // 2, self.flattened_size)
        
        ## decoder layers ##
        self.t_conv1 = nn.ConvTranspose1d(64, 32, kernel_size=2, stride=2)
        self.t_conv2 = nn.ConvTranspose1d(32, 16, kernel_size=2, stride=2)
        self.conv_out = nn.Conv1d(16, 1, kernel_size=3, padding=1)

    def forward(self, x):
        ## encode ##
        z = self.pool1(F.elu(self.bn1(self.conv1(x))))  # ELU for better negative handling
        z = self.pool2(F.elu(self.bn2(self.conv2(z))))
        z = self.dropout(z)
        
        # Bottleneck
        batch_size = z.size(0)
        z_flat = z.view(batch_size, -1)
        z_compressed = F.elu(self.fc(z_flat))
        z_latent = z_compressed.clone()
        
        # Decode
        z_expanded = F.elu(self.fc_decode(z_compressed))
        z_unflat = z_expanded.view(batch_size, 64, self.encoded_length)
        
        ## decode ##
        x = F.elu(self.t_conv1(z_unflat))
        x = F.elu(self.t_conv2(x))
        x = self.conv_out(x)  # No activation for reconstruction
        
        # Use adaptive interpolation to match target output size
        if x.size(-1) != self.output_size:
            x = F.interpolate(x, size=self.output_size, mode='linear', align_corners=False)
        
        return x, z_latent

class LinearDenoiser(nn.Module):
    def __init__(self, input_size=33, output_size=32):
        super(LinearDenoiser, self).__init__()

        self.input_size = input_size
        self.output_size = output_size
        
        ## encoder layers ##
        self.linear1 = nn.Linear(input_size, 32)
        self.bn1 = nn.BatchNorm1d(32)
        self.dropout1 = nn.Dropout(0.2)
        
        self.linear2 = nn.Linear(32, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.dropout2 = nn.Dropout(0.2)
        
        # Bottleneck layer
        self.linear3 = nn.Linear(64, 32)
        self.bn3 = nn.BatchNorm1d(32)
        
        ## decoder layers ##
        self.linear4 = nn.Linear(32, 64)
        self.bn4 = nn.BatchNorm1d(64)
        self.dropout3 = nn.Dropout(0.2)
        
        self.linear5 = nn.Linear(64, 128)
        self.bn5 = nn.BatchNorm1d(128)
        self.dropout4 = nn.Dropout(0.2)
        
        self.linear_out = nn.Linear(128, output_size)
        
    def forward(self, x):
        if len(x.shape) == 3:
            x = x.squeeze(1)
        
        ## encoder ##
        x = F.gelu(self.bn1(self.linear1(x)))  # GELU works well with normalized data
        x = self.dropout1(x)
        
        x = F.gelu(self.bn2(self.linear2(x)))
        x = self.dropout2(x)
        
        z_latent = x.clone()

        # Bottleneck
        encoded = F.gelu(self.bn3(self.linear3(x)))
        
        ## decoder ##
        x = F.gelu(self.bn4(self.linear4(encoded)))
        x = self.dropout3(x)
        
        x = F.gelu(self.bn5(self.linear5(x)))
        x = self.dropout4(x)
        
        # Output layer (no activation for reconstruction)
        x = self.linear_out(x)
        x = x.unsqueeze(1)
        
        return x, z_latent
    
class Classifier(nn.Module):
    def __init__(self, input_length=33, num_classes=4):
        super(Classifier, self).__init__()
        
        # Use adaptive pooling for cleaner size handling
        self.conv1 = nn.Conv1d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(32)
        self.pool1 = nn.AdaptiveMaxPool1d(input_length // 2)
        
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(64)
        self.pool2 = nn.AdaptiveMaxPool1d(input_length // 4)
        
        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(128)
        self.pool3 = nn.AdaptiveMaxPool1d(input_length // 8)
        
        # Global average pooling to handle any input size
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # Fully connected layers
        self.fc1 = nn.Linear(128, 64)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, z):
        z = self.pool1(F.elu(self.bn1(self.conv1(z))))  # ELU for conv layers
        z = self.pool2(F.elu(self.bn2(self.conv2(z))))
        z = self.pool3(F.elu(self.bn3(self.conv3(z))))
        
        z = self.global_pool(z)  # Global pooling to size [batch, 128, 1]
        z = z.view(z.size(0), -1)  # Flatten to [batch, 128]
        
        z = F.gelu(self.fc1(z))  # GELU for FC layers
        z = self.dropout(z)
        z = self.fc2(z)  # No activation - CrossEntropyLoss expects logits

        return z
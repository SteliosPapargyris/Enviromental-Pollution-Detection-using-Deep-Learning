# 🌍 Environmental Pollution Detection using Deep Learning
## 📌 Project Description
This project utilizes deep learning techniques to detect and classify environmental pollution. The model is designed to analyze gas pollution

## 🔥 Key Features


## ⚡ Installation
To set up the project locally, follow these steps:
```bash
git clone https://github.com/SteliosPapargyris/Environmental-Pollution-Detection-using-Deep-Learning.git
```
```bash
cd Environmental-Pollution-Detection-using-Deep-Learning
```
```bash
pip install -r requirements.txt
```

## 🚀 Usage

Run the main script:
```bash
python main.py
```
Run tests:

```bash
python test.py
```

## 🛠 Project Structure
```bash
📂 Environmental-Pollution-Detection-using-Deep-Learning
│── 📂 pths/        # Model weights
│── 📂 utils/       # Helper scripts and utility functions
│── 📂 out/         # Output results
│── main.py        # Main execution script
│── test.py        # Unit tests for model evaluation
│── requirements.txt  # Dependencies
```

## 📢 Release Notes

### 🚀 Upcoming Changes
- Add from Chip 5 to training (10%, then 20% etc until test accuracy is high)
- .clone --> in latent space of encoder of z
- Write the thesis

For future:
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(64)
        self.pool2 = nn.MaxPool1d(2)
        # Lout = (16 + 2 - 3)/1 + 1 --> Lout = 16
        # pooling --> 8

        # input length --> 32
        # After pool1: 16, After pool2: 8, 64 channels
        self.flattened_size = 64 * 8  # update based on input size
pool2: from 8 to 6 
Drop encoder to 8 bits.

### ✅ Implemented

🔹 v1.4 (May 2025)
- Added Temperature as a feature to classifier
- Changed unseen chips and did all the combinations for train and test chips and of course the corresponding baseline Chips. Best results for Train Chips 1,2,3,4 and Test Chip 5 and baseline Chip 4
- 
🔹 v1.3 (April 2025)
- 🛠 Added a **dense layer after Conv2** in the convolutional denoiser for improved feature extraction
- Shuffle dataset (e.g chip 1, 3, 4 shuffled) and the target chip will be chip 2. Not necessary to do continual learning.

🔹 v1.2 (March 2025)
- 🏷️ Standardizing all chips with autoencoders to follow the structure of **Chip 1** (chosen as the reference chip)

🔹 v1.1 (March 2025)
- ⚡ Optimized deep learning model for faster inference
- 🛠 Improved dataset pre-processing pipeline
- 🐞 Bug fixes in test cases

🔹 v1.0 (February 2025)
- 🚀 Initial release with baseline deep learning model
- 🏗️ Added dataset preprocessing and augmentation
- 🧠 Implemented pollution classification with CNN

## 🤝 Contributing
Contributions are welcome! Feel free to fork the repo, submit issues, or create pull requests.

## 📜 License
This project is licensed under the MIT License.


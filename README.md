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
- Take **ideas** from other **papers** (for let's say autoencoder -> xeR^32 -> x anhkei R^32
- Start with describing the dataset (ask for the paper)
- Auto Encoder describe z latent space, we use it in the decoder (figure) mathematically and visually + Normalization in input (like callibration), based on class 4 tihs mathematical expression (mean and std of class 4 to other classes in the same chip)
- After that just say that i have this classifier
- In proposed Method i should be as abstract as possible. And then in a different chapter more detail to every
- What is the proposed method -> encoder-decoder, after that Global Local Model (Feature Extraction)
- I should compare "same" models with same features not a model that has a temperature feature with one that does not
- Proposed Method -> auto encoder with 32 features --> % accuracy
- CNN --> % accuracy
- etc
- Proposed Method -> auto encoder with 33 features --> % accuracy (bigger accuracy with temperature) not compare with cnn or global and local with 32 features
- In a **table** i should compare those different methods (must be clear)
- Class 4 Mean and subtract
other (not class 4) classes
with this mean of class 4
100.0% 48.12% (It's a part of autoencoder implementation so it must be into thesis)
- Normalization in plots not with numbers. Numbers only for our results

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

🔹 v1.5 (May 2025)
- Implemented project with 20 chips and 100 chips also
- Added percentages of test chip in training and remove those samples of data from inference
- .clone in latent space of encoder of z



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


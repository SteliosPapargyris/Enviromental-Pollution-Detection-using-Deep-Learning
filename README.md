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

Run the autoencoder and classifier and test script:
```bash
python generate_interferogram.py && python create_csv_dataset.py && python normalize_chips.py && python train_autoencoder.py && python train_classifier.py
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

- Normalization to all samples not excluding Class 4 and in test set i know only the statistics (e.g mean and std) not the class of the samples

- Few shot learning (fine tune, started incrementally from 10%) + table if not do normalization, if not do autoencoder, if not do both etc -> see the improvement of the accuracy

- t sne algorithm to output of decoder and latent space to see if they are being διαχωριζονται

no chip 1,2,3,4,5 all in one and then normalization, autoencoder and inference
Normalization 
One autoencoder for every chip and then target for autoencoder will be the sample of T= (e.g = 25C) 
One more autoencoder that will take the above outputed data to one baseline chip (initial before 1st autoencoder for T=25C) (selection is up to me) 
and then classifier 

In test set of the initial splitting of dataset (70-20-10)

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


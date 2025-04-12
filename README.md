# Environmental Pollution Detection using Deep Learning

This repository explores different implementations for detecting environmental pollution from satellite imagery, with a focus on **continual learning** and **unsupervised representation learning**.

---

## 🧪 Current Experiment: Shuffled Dataset + Autoencoders

### 🔄 Dataset Restructuring

Originally, the dataset was split by **chips**:

- **Training Set:** Chips 1 to 4  
- **Test Set:** Chip 5

In this experimental branch, we **shuffled** the dataset:

- Each training chip (e.g., chip 1) now includes a mix of samples from **chips 1 to 4**.
- The **test set remains chip 5** to evaluate performance on unseen data.
- This was done to remove chip-specific biases and force models to generalize better.

---

### 🧠 Continual Learning with Autoencoders

The main goal of this stage was to explore **continual learning** using **autoencoders**.

#### Objectives:
- Perform **denoising** on the training samples.
- Use autoencoders as a stepping stone for **unsupervised domain adaptation** and **feature learning** across chips.

#### Approach:
- Trained an autoencoder incrementally across shuffled chips.
- Autoencoders were expected to learn meaningful latent representations and remove noise.

---

### ⚠️ Key Issue Identified

Despite the reshuffling and setup, this implementation **did not succeed** in the intended denoising or domain adaptation task. Why?

1. **Distribution Mismatch**:
   - The autoencoder was not tuned or trained to **match the distribution** of the target domain (chip 2).
   - Data from all chips was used without domain normalization or alignment.

2. **Noisy, Mixed Domain Inputs**:
   - Without controlling for domain shifts between chips, the autoencoder had to learn across heterogeneous data.
   - This led to poor reconstructions and ineffective denoising.

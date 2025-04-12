# Autoencoder - Matched to Chip 1

This branch uses an autoencoder to align the distribution of chips 1–4 to that of chip 1.  
All training samples are passed through the trained autoencoder to normalize them toward chip 1’s domain.  
This serves as a preprocessing step for denoising and domain unification before downstream continual learning.

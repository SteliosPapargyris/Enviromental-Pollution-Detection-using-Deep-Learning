# Autoencoder - Matched to Chip 2

This branch uses an autoencoder to align the distribution of chips 1–4 to that of chip 2.  
All training samples are passed through the trained autoencoder to normalize them toward chip 1=2’s domain.  
This serves as a preprocessing step for denoising and domain unification before downstream continual learning.

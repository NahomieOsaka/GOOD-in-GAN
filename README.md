# GOOD-in-GAN

Apprendre les GANs appliqués aux événements extrêmes (ici : pertes extrêmes financières).

Cette branche contient des notebooks d'initiation en PyTorch, des utilitaires et un environnement pour commencer depuis zéro.

Structure ajoutée dans cette branche `add/setup-notebooks` :

- environment.yml : environnement conda/pip recommandé
- notebooks/
  - 00_setup.ipynb : installation, vérification GPU, guide pas-à-pas
  - 01_toy_gan.ipynb : GAN 1D (PyTorch) pour apprendre une distribution heavy-tailed (pertes)
- src/
  - models.py : architectures Generator / Discriminator simples
  - data_utils.py : génération de données Pareto / téléchargement S&P500 via yfinance
  - train.py : boucle d'entraînement simple pour le GAN
  - eval_extremes.py : fonctions d'évaluation pour quantiles et tail index
- .gitignore

But : tu m'as demandé d'utiliser un dataset financier public pour l'exemple (option B). Le notebook 01 télécharge automatiquement des returns S&P500 via yfinance et montre comment prétraiter pour extraire pertes extrêmes.

Si tu veux que je pousse d'autres exemples (DCGAN, WGAN-GP, cGAN) je peux les ajouter étape par étape.

Guide rapide :
1) Crée un environnement conda : `conda env create -f environment.yml`
2) Active-le : `conda activate good-in-gan-env`
3) Lance `jupyter lab` et ouvre `notebooks/01_toy_gan.ipynb`

Tout est commenté en français et pensé pour un débutant.

"""Fichier utilitaires modèles simple pour GANs 1D"""

import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, z_dim=16, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    def forward(self, z):
        return self.net(z).squeeze(-1)

class Discriminator(nn.Module):
    def __init__(self, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        x = x.unsqueeze(-1).float()
        return self.net(x).squeeze(-1)

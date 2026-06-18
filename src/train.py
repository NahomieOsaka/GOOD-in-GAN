"""Script simple pour entraîner un GAN - utilitaire (utilise les modèles de src.models)"""

import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim
import torch.nn as nn
import numpy as np

from src.models import Generator, Discriminator


def train_gan(data_array, z_dim=16, n_epochs=50, batch_size=128, lr=1e-4, device=None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    X = torch.tensor(data_array, dtype=torch.float32).unsqueeze(-1)
    loader = DataLoader(TensorDataset(X), batch_size=batch_size, shuffle=True)

    G = Generator(z_dim=z_dim).to(device)
    D = Discriminator().to(device)
    opt_G = optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.9))
    opt_D = optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.9))
    loss_fn = nn.BCELoss()
    real_label = 1.0
    fake_label = 0.0

    history = {'d_loss': [], 'g_loss': []}
    for epoch in range(n_epochs):
        d_losses = []
        g_losses = []
        for batch in loader:
            real = batch[0].to(device)
            bs = real.size(0)

            D.zero_grad()
            labels_real = torch.full((bs,), real_label, device=device)
            output_real = D(real).view(-1)
            loss_real = loss_fn(output_real, labels_real)

            z = torch.randn(bs, z_dim, device=device)
            fake = G(z).detach()
            labels_fake = torch.full((bs,), fake_label, device=device)
            output_fake = D(fake).view(-1)
            loss_fake = loss_fn(output_fake, labels_fake)

            loss_D = loss_real + loss_fake
            loss_D.backward()
            opt_D.step()

            G.zero_grad()
            z = torch.randn(bs, z_dim, device=device)
            gen = G(z)
            labels_for_g = torch.full((bs,), real_label, device=device)
            output = D(gen).view(-1)
            loss_G = loss_fn(output, labels_for_g)
            loss_G.backward()
            opt_G.step()

            d_losses.append(loss_D.item())
            g_losses.append(loss_G.item())

        history['d_loss'].append(np.mean(d_losses))
        history['g_loss'].append(np.mean(g_losses))
        print(f'Epoch {epoch+1}/{n_epochs} D_loss={history["d_loss"][-1]:.4f} G_loss={history["g_loss"][-1]:.4f}')

    return G, D, history

"""Utilitaires pour données : samples Pareto et téléchargement S&P500"""

import numpy as np
import pandas as pd

def sample_pareto(size, xm=1.0, alpha=2.5, seed=None):
    """Retourne des échantillons de loi de Pareto (positifs)
    xm : scale (seuil minimal), alpha : shape
    """
    if seed is not None:
        np.random.seed(seed)
    u = np.random.rand(size)
    return xm / (u ** (1.0 / alpha))


def download_sp500_returns(start='2000-01-01', end=None, ticker='^GSPC'):
    """Télécharge les prix S&P500 via yfinance et retourne les pertes (returns négatifs).
    Retourne une série pandas de pertes positives (abs of negative returns) si applicable.
    """
    import yfinance as yf
    data = yf.download(ticker, start=start, end=end)
    data = data['Adj Close'].dropna()
    returns = data.pct_change().dropna()
    # pertes positives = -returns when returns < 0
    losses = (-returns[returns < 0]).dropna()
    losses.name = 'loss'
    return losses

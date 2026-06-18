"""Évaluation pour événements extrêmes : quantiles, QQ-plot et estimateur de Hill"""

import numpy as np
import matplotlib.pyplot as plt


def compare_quantiles(real, gen, qs=[0.9, 0.95, 0.99, 0.999]):
    real = np.asarray(real)
    gen = np.asarray(gen)
    out = {}
    for q in qs:
        out[q] = {'real': np.quantile(real, q), 'gen': np.quantile(gen, q)}
    return out


def plot_qq_tail(real, gen, p_min=0.9, p_max=0.999, n=100):
    p = np.linspace(p_min, p_max, n)
    real_q = np.quantile(real, p)
    gen_q = np.quantile(gen, p)
    plt.figure(figsize=(6,6))
    plt.scatter(real_q, gen_q, s=10)
    plt.plot([real_q.min(), real_q.max()], [real_q.min(), real_q.max()], 'r--')
    plt.xlabel('Quantiles réels (queue)')
    plt.ylabel('Quantiles générés (queue)')
    plt.title(f'QQ-plot (p in [{p_min},{p_max}])')
    plt.show()


def hill_estimator(samples, k=100):
    """Estimateur de Hill pour l'indice de queue. samples doit être trié desc ou non, on trie.
    k : nombre de top order statistics à utiliser (taille de la queue)
    Retourne gamma (tail index) et standard error approximatif.
    """
    x = np.sort(np.asarray(samples))
    n = len(x)
    # on s'intéresse aux k plus grands
    if k <= 0 or k >= n:
        raise ValueError('k doit être entre 1 et n-1')
    x_tail = x[-k:]
    x_k = x[-k-1]
    # Hill estimator (pour Pareto) : gamma_hat = (1/k) * sum(log(x_i) - log(x_k))
    logs = np.log(x_tail) - np.log(x_k)
    gamma = np.mean(logs)
    se = np.std(logs) / np.sqrt(k)
    return gamma, se

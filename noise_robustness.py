# Noise Robustness Analysis: critical visibility under depolarizing noise

# # Noise Robustness Analysis
# 
# Depolarizing noise model: ρ_noisy = v·ρ + (1−v)·I/d
# 
# Critical visibility: v* = (α − n/d) / (η_d − n/d)
# 
# Contextuality is certified when v > v*.

import numpy as np

def critical_visibility(alpha, eta_d, n, d):
    """Kritik görünürlük: depolarizing noise modeli."""
    noise_val = n / d
    if eta_d <= alpha:
        return float('inf')  # kuantum avantajı yok
    return (alpha - noise_val) / (eta_d - noise_val)

def noise_tolerance(v_star):
    """Maksimum tolere edilebilir gürültü = 1 - v*"""
    return 1 - v_star

# Graflar ve değerleri
graphs = [
    # (isim,  n,  alpha, eta_d,          d,   not)
    ('C5 (KCBS)',      5, 2, 2.23607,  3, 'eta_3 = sqrt(5)'),
    ('C7',             7, 3, 3.31767,  3, 'eta_3 analitik'),
    ('Quad-C5 (d=3)',  8, 3, 3.23607,  3, 'eta_3 = 1+sqrt(5)'),
    ('Quad-C5 (d=4)',  8, 3, 3.46784,  4, 'eta_4 ~ theta'),
    ('Wagner (d=4)',   8, 3, 3.41421,  4, 'eta_4 ~ theta'),
]

print(f'{"Graf":<22} {"n":<4} {"α":<4} {"η_d":<8} {"d":<4} {"v*":<8} {"1-v* (gürültü tol.)":<22} {"Not"}')
print('-' * 95)

for name, n, alpha, eta_d, d, note in graphs:
    vs = critical_visibility(alpha, eta_d, n, d)
    nt = noise_tolerance(vs)
    print(f'{name:<22} {n:<4} {alpha:<4} {eta_d:<8.5f} {d:<4} {vs:<8.4f} {nt:<22.4f} {note}')

# Quad-C5 vs Wagner karşılaştırması
print('=== Quad-C5 vs Wagner (d=4) ===')
vs_qc = critical_visibility(3, 3.46784, 8, 4)
vs_wg = critical_visibility(3, 3.41421, 8, 4)
print(f'Quad-C5: v* = {vs_qc:.6f}  →  gürültü toleransı = {1-vs_qc:.6f}')
print(f'Wagner:  v* = {vs_wg:.6f}  →  gürültü toleransı = {1-vs_wg:.6f}')
print(f'Fark: Quad-C5, Wagner\'den {(vs_wg-vs_qc)*100:.2f}% daha az görünürlük gerektiriyor')
print()

# d=3 karşılaştırması
print('=== Quad-C5 vs KCBS (d=3) ===')
vs_qc3 = critical_visibility(3, 3.23607, 8, 3)
vs_kcbs = critical_visibility(2, 2.23607, 5, 3)
print(f'Quad-C5 (d=3): v* = {vs_qc3:.6f}  →  gürültü toleransı = {1-vs_qc3:.6f}')
print(f'KCBS    (d=3): v* = {vs_kcbs:.6f}  →  gürültü toleransı = {1-vs_kcbs:.6f}')
print()
print('Analitik kontrol:')
print(f'  Quad-C5: (3 - 8/3) / (1+sqrt(5) - 8/3) = {(3-8/3)/(1+5**0.5-8/3):.6f}')
print(f'  KCBS:    (2 - 5/3) / (sqrt(5) - 5/3)   = {(2-5/3)/(5**0.5-5/3):.6f}')

import matplotlib.pyplot as plt
import numpy as np

v_range = np.linspace(0.5, 1.0, 500)

def noisy_val(v, eta_d, n, d):
    return v * eta_d + (1-v) * n/d

configs = [
    ('KCBS (d=3)',     2.23607, 5, 3, 2, '#2ecc71',  '--'),
    ('C7 (d=3)',       3.31767, 7, 3, 3, '#3498db',  '--'),
    ('Quad-C5 (d=3)',  3.23607, 8, 3, 3, '#e74c3c',  '-'),
    ('Quad-C5 (d=4)',  3.46784, 8, 4, 3, '#c0392b',  '-'),
    ('Wagner (d=4)',   3.41421, 8, 4, 3, '#f39c12',  '--'),
]

fig, ax = plt.subplots(figsize=(9, 5))

for name, eta_d, n, d, alpha, color, ls in configs:
    vals = noisy_val(v_range, eta_d, n, d)
    ax.plot(v_range, vals, color=color, ls=ls, lw=2, label=name)
    vs = critical_visibility(alpha, eta_d, n, d)
    ax.axvline(vs, color=color, lw=0.8, alpha=0.4)

ax.axhline(3, color='black', lw=1.5, ls=':', label='Klasik sınır α=3 (n=8)')
ax.axhline(2, color='gray',  lw=1.5, ls=':', label='Klasik sınır α=2 (KCBS)')

ax.set_xlabel('Görünürlük v', fontsize=13)
ax.set_ylabel('S(v) = v·η_d + (1-v)·n/d', fontsize=13)
ax.set_title('Gürültü Altında Kuantum Değer', fontsize=13)
ax.legend(fontsize=9, loc='upper left')
ax.set_xlim(0.5, 1.0)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('data/noise_robustness.png', dpi=150)
plt.show()
print('✓ data/noise_robustness.png kaydedildi')


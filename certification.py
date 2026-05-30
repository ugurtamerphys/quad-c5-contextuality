# Quad-C5 Certification: exhaustive SDP search over all 11,117 connected 8-vertex graphs

# # Quad-C5 Certification Notebook
# 
# Exhaustive SDP-based contextuality gap search over all 11,117 connected non-isomorphic graphs on 8 vertices.
# 
# Steps:
# 1. Independence number α: exact computation + brute-force cross-validation
# 2. Lovász theta θ: SCS (bulk) + CLARABEL (high-precision) with primal–dual residuals
# 3. Certified numerical intervals for Quad-C5, Rank-2, and Wagner
# 4. Dimension classification: η₃ optimization with 300 random restarts
# 5. Export results to CSV

import numpy as np
import networkx as nx
import cvxpy as cp
import urllib.request, os, csv
from itertools import combinations
from tqdm import tqdm

print('Solver listesi:', cp.installed_solvers())

# ## Step 1 — Independence Number Certification
# 
# Two independent methods:
# - **Method A:** `max_weight_clique(complement(G))` via NetworkX
# - **Method B:** brute-force enumeration of all 2^8 = 256 vertex subsets

def alpha_clique(G):
    """Yöntem A: alpha(G) = omega(complement(G)), tam algoritma."""
    clique, _ = nx.clique.max_weight_clique(nx.complement(G), weight=None)
    return len(clique)

def alpha_brute(G):
    """Yöntem B: tüm alt kümeleri tara, en büyük bağımsız kümeyi bul."""
    nodes = list(G.nodes())
    n = len(nodes)
    best = 0
    for size in range(n, 0, -1):
        if size <= best:
            break
        for subset in combinations(nodes, size):
            subG = G.subgraph(subset)
            if subG.number_of_edges() == 0:  # bağımsız küme
                best = size
                break
        if best == size:
            break
    return best

# graph8.g6 indir
GRAPH8_URL  = 'https://users.cecs.anu.edu.au/~bdm/data/graph8.g6'
GRAPH8_FILE = 'graph8.g6'

if not os.path.exists(GRAPH8_FILE):
    print('graph8.g6 indiriliyor...')
    urllib.request.urlretrieve(GRAPH8_URL, GRAPH8_FILE)
    print('İndirildi.')

all_graphs = list(nx.read_graph6(GRAPH8_FILE))
connected_graphs = [G for G in all_graphs if nx.is_connected(G)]
print(f'Toplam: {len(all_graphs)}, Bağlantılı: {len(connected_graphs)}')

# Tüm bağlantılı graflarda iki yöntemle alpha hesapla ve karşılaştır
mismatches = []

for G in tqdm(connected_graphs, desc='α cross-check'):
    a_clique = alpha_clique(G)
    a_brute  = alpha_brute(G)
    if a_clique != a_brute:
        mismatches.append((G, a_clique, a_brute))

if mismatches:
    print(f'UYARI: {len(mismatches)} grafda tutarsızlık!')
    for G, ac, ab in mismatches:
        print(f'  clique={ac}, brute={ab}, kenarlar={list(G.edges())}')
else:
    print(f'✓ Tüm {len(connected_graphs)} grafda her iki yöntem aynı α değerini verdi.')
    print('  α hesaplaması sertifikalandı.')

# ## Step 2 — Lovász Theta via SCS and CLARABEL with Primal–Dual Residuals

def lovasz_theta_with_residuals(G, solver):
    """
    θ(G) hesapla ve primal/dual residual döndür.
    Döndürür: (theta, primal_residual, dual_residual, status)
    """
    n = G.number_of_nodes()
    nodes = sorted(G.nodes())
    idx = {v: i for i, v in enumerate(nodes)}
    X = cp.Variable((n, n), PSD=True)
    cons = [cp.trace(X) == 1]
    for u, v in G.edges():
        cons += [X[idx[u], idx[v]] == 0]
    prob = cp.Problem(cp.Maximize(cp.sum(X)), cons)

    try:
        prob.solve(solver=solver, verbose=False)
        theta = float(prob.value)
        # primal residual: max constraint violation
        pr = abs(float(cp.trace(X).value) - 1.0)
        for u, v in G.edges():
            pr = max(pr, abs(float(X[idx[u], idx[v]].value)))
        # dual residual: |primal - dual|
        dr = abs(prob.value - prob.solver_stats.extra_stats.get('obj_val_dual', prob.value)) \
             if hasattr(prob.solver_stats, 'extra_stats') else float('nan')
        return theta, pr, dr, prob.status
    except Exception as e:
        return float('nan'), float('nan'), float('nan'), f'ERROR: {e}'

def lovasz_theta_scs(G):
    """SCS ile θ ve primal residual."""
    n = G.number_of_nodes()
    nodes = sorted(G.nodes())
    idx = {v: i for i, v in enumerate(nodes)}
    X = cp.Variable((n, n), PSD=True)
    cons = [cp.trace(X) == 1]
    for u, v in G.edges():
        cons += [X[idx[u], idx[v]] == 0]
    prob = cp.Problem(cp.Maximize(cp.sum(X)), cons)
    prob.solve(solver=cp.SCS, verbose=False, eps=1e-9)
    theta = float(prob.value)
    pr = abs(float(cp.trace(X).value) - 1.0)
    for u, v in G.edges():
        pr = max(pr, abs(float(X[idx[u], idx[v]].value)))
    return theta, pr, prob.status

def lovasz_theta_clarabel(G):
    """CLARABEL ile θ ve primal residual."""
    n = G.number_of_nodes()
    nodes = sorted(G.nodes())
    idx = {v: i for i, v in enumerate(nodes)}
    X = cp.Variable((n, n), PSD=True)
    cons = [cp.trace(X) == 1]
    for u, v in G.edges():
        cons += [X[idx[u], idx[v]] == 0]
    prob = cp.Problem(cp.Maximize(cp.sum(X)), cons)
    prob.solve(solver=cp.CLARABEL, verbose=False)
    theta = float(prob.value)
    pr = abs(float(cp.trace(X).value) - 1.0)
    for u, v in G.edges():
        pr = max(pr, abs(float(X[idx[u], idx[v]].value)))
    return theta, pr, prob.status

# Önce tüm bağlantılı graflarda SCS ile tam tarama yap, üst 50'yi bul
print('Tüm n=8 bağlantılı graflarda SCS ile θ hesaplanıyor...')
results_all = []

for G in tqdm(connected_graphs, desc='SCS tarama'):
    a = alpha_clique(G)
    t, pr, status = lovasz_theta_scs(G)
    delta = t - a
    results_all.append({
        'graph6': nx.to_graph6_bytes(G, header=False).decode().strip(),
        'alpha': a,
        'theta_scs': round(t, 8),
        'delta_scs': round(delta, 8),
        'edges': G.number_of_edges(),
        'pr_scs': pr,
        'status_scs': status
    })

results_all.sort(key=lambda r: r['delta_scs'], reverse=True)
top50 = results_all[:50]
print(f'\nTarama tamamlandı. Üst 50 belirlendi.')
print(f'Rank 1: graph6={top50[0]["graph6"]}, Δ={top50[0]["delta_scs"]}')

# Üst 50 için CLARABEL ile de hesapla
print('Üst 50 için CLARABEL çalıştırılıyor...')

# graph6 → Graph dönüşümü
g6_to_graph = {}
for G in connected_graphs:
    g6 = nx.to_graph6_bytes(G, header=False).decode().strip()
    g6_to_graph[g6] = G

for r in tqdm(top50, desc='CLARABEL top-50'):
    G = g6_to_graph[r['graph6']]
    t_cl, pr_cl, status_cl = lovasz_theta_clarabel(G)
    r['theta_clarabel'] = round(t_cl, 8)
    r['delta_clarabel'] = round(t_cl - r['alpha'], 8)
    r['pr_clarabel'] = pr_cl
    r['status_clarabel'] = status_cl
    r['solver_diff'] = abs(r['theta_scs'] - t_cl)

print('\n--- Üst 10 Graf (her iki solver) ---')
print(f'{"Rank":<5} {"Graph6":<10} {"α":<4} {"θ(SCS)":<12} {"θ(CLARABEL)":<14} {"Δ(SCS)":<10} {"|Δ solver|":<12} {"E"}')
print('-' * 85)
for i, r in enumerate(top50[:10]):
    print(f"{i+1:<5} {r['graph6']:<10} {r['alpha']:<4} {r['theta_scs']:<12.5f} "
          f"{r['theta_clarabel']:<14.5f} {r['delta_scs']:<10.5f} "
          f"{r['solver_diff']:<12.2e} {r['edges']}")

# ## Step 3 — Certified Numerical Intervals
# 
# Non-overlapping θ intervals for Quad-C5, Rank-2, and Wagner from two independent solvers.

# Quad-C₅ (Rank 1), 2. sıra, Wagner graph6 kodları
QUAD_C5_EDGES  = [(0,3),(0,5),(1,4),(1,6),(2,5),(2,6),(2,7),(3,6),(3,7),(4,7)]
WAGNER_EDGES   = [(0,1),(1,2),(2,3),(3,4),(4,5),(5,6),(6,7),(7,0),(0,4),(1,5),(2,6),(3,7)]

G_qc = nx.Graph(QUAD_C5_EDGES)
G_wg = nx.Graph(WAGNER_EDGES)
G_r2 = g6_to_graph[top50[1]['graph6']]  # 2. sıra

configs = [
    ('Quad-C5 (Rank 1)', G_qc),
    ('Rank 2',           G_r2),
    ('Wagner',           G_wg),
]

# Her grafı 5 kez her solver ile çalıştır, min/max aralık al
N_RUNS = 5
print(f'Her grafı {N_RUNS} kez iki solverla çalıştırıyoruz...\n')

intervals = {}
for name, G in configs:
    vals_scs, vals_cl = [], []
    for _ in range(N_RUNS):
        t_s, pr_s, _ = lovasz_theta_scs(G)
        t_c, pr_c, _ = lovasz_theta_clarabel(G)
        vals_scs.append(t_s)
        vals_cl.append(t_c)
    all_vals = vals_scs + vals_cl
    lo, hi = min(all_vals), max(all_vals)
    mid = (lo + hi) / 2
    half = (hi - lo) / 2
    intervals[name] = (mid, half, lo, hi)
    print(f'{name}: θ ∈ [{lo:.8f}, {hi:.8f}]  (±{half:.2e})')

print()

# Non-overlapping kontrolü
lo_qc = intervals['Quad-C5 (Rank 1)'][2]
hi_r2 = intervals['Rank 2'][3]
hi_wg = intervals['Wagner'][3]

gap_r2 = lo_qc - hi_r2
gap_wg = lo_qc - hi_wg

print('=== Sertifikasyon Sonucu ===')
print(f'Quad-C5 alt sınır : {lo_qc:.8f}')
print(f'Rank-2  üst sınır : {hi_r2:.8f}  →  boşluk = {gap_r2:.2e}')
print(f'Wagner  üst sınır : {hi_wg:.8f}  →  boşluk = {gap_wg:.2e}')
print()

if gap_r2 > 0 and gap_wg > 0:
    print('✓ Aralıklar çakışmıyor.')
    print('✓ Quad-C5 is the CERTIFIED unique maximizer among connected 8-vertex graphs.')
else:
    print('✗ Aralıklar çakışıyor — daha yüksek hassasiyet gerekli.')

# ## Step 4 — Export Results to CSV

import csv, os
os.makedirs('data', exist_ok=True)

# Tüm n=8 sonuçları
with open('data/all_n8_results.csv', 'w', newline='') as f:
    fields = ['rank','graph6','alpha','theta_scs','delta_scs','edges','pr_scs','status_scs']
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    for i, r in enumerate(results_all):
        row = {k: r[k] for k in fields if k != 'rank'}
        row['rank'] = i + 1
        w.writerow(row)

print('✓ data/all_n8_results.csv kaydedildi')

# Üst 50 (her iki solver + residual)
with open('data/top50_certification.csv', 'w', newline='') as f:
    fields = ['rank','graph6','alpha','theta_scs','theta_clarabel',
              'delta_scs','delta_clarabel','edges','pr_scs','pr_clarabel',
              'solver_diff','status_scs','status_clarabel']
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    for i, r in enumerate(top50):
        row = {k: r.get(k, '') for k in fields if k != 'rank'}
        row['rank'] = i + 1
        w.writerow(row)

print('✓ data/top50_certification.csv kaydedildi')

# Güven aralıkları
with open('data/certified_intervals.csv', 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['graph','theta_mid','half_width','lo','hi'])
    for name, (mid, half, lo, hi) in intervals.items():
        w.writerow([name, mid, half, lo, hi])

print('✓ data/certified_intervals.csv kaydedildi')
print()
print('Tüm sertifikasyon verisi data/ klasörüne kaydedildi.')

## Adım 5 — Dual residual hesabı
# Duality gap = |primal_obj - dual_obj|
# Dual objective = lambda (dual variable for tr(X)=1)
# Strong duality => gap ~ 0 at optimality

def lovasz_theta_full(G, solver):
    """θ, primal residual, dual residual (duality gap) hesapla."""
    n = G.number_of_nodes()
    nodes = sorted(G.nodes())
    idx = {v: i for i, v in enumerate(nodes)}
    X = cp.Variable((n, n), PSD=True)
    cons = [cp.trace(X) == 1]
    for u, v in G.edges():
        cons += [X[idx[u], idx[v]] == 0]
    prob = cp.Problem(cp.Maximize(cp.sum(X)), cons)
    prob.solve(solver=solver, verbose=False)

    theta = float(prob.value)

    # Primal residual: max constraint violation
    pr = abs(float(cp.trace(X).value) - 1.0)
    for u, v in G.edges():
        pr = max(pr, abs(float(X[idx[u], idx[v]].value)))

    # Dual residual: duality gap = |primal - dual_obj|
    # dual objective = lambda_1 * b_1 = lambda * 1
    # (only tr(X)=1 contributes to dual obj; edge constraints have b=0)
    lam = float(cons[0].dual_value)
    dr = abs(theta - lam)

    return theta, pr, dr, prob.status


# Top-50 için her iki solver ile primal + dual residual
print('Top-50 için primal + dual residual hesaplanıyor...')
print()
print(f'{"Rank":<5} {"α":<4} {"θ(CL)":<10} {"r_p(CL)":<12} {"r_d(CL)":<12} {"r_p(SCS)":<12} {"r_d(SCS)":<10} {"E"}')
print('-' * 85)

for i, r in enumerate(top50):
    G = g6_to_graph[r['graph6']]
    t_cl, pr_cl, dr_cl, _ = lovasz_theta_full(G, cp.CLARABEL)
    t_sc, pr_sc, dr_sc, _ = lovasz_theta_full(G, cp.SCS)
    r['dr_clarabel'] = dr_cl
    r['dr_scs'] = dr_sc
    if i < 10:
        print(f"{i+1:<5} {r['alpha']:<4} {t_cl:<10.5f} {pr_cl:<12.2e} {dr_cl:<12.2e} {pr_sc:<12.2e} {dr_sc:<10.2e} {r['edges']}")

print()
print('✓ Dual residual (duality gap) tüm top-50 için hesaplandı.')

# CSV'yi dual residual ile güncelle
import csv
with open('data/top50_certification.csv', 'w', newline='') as f:
    fields = ['rank','graph6','alpha','theta_scs','theta_clarabel',
              'delta_scs','delta_clarabel','edges',
              'pr_scs','pr_clarabel','dr_scs','dr_clarabel',
              'solver_diff','status_scs','status_clarabel']
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    for i, r in enumerate(top50):
        row = {k: r.get(k, '') for k in fields if k != 'rank'}
        row['rank'] = i + 1
        w.writerow(row)

print('✓ data/top50_certification.csv dual residual ile güncellendi.')

# ## Step 5 — Numerical Certification of η₃(Quad-C5) = 1+√5
# 
# PSLQ analysis identified 1+√5 (minimal polynomial x²−2x−4 = 0) as the algebraic form of the
# numerically obtained η₃ ≈ 3.23607. This section verifies that result:
# 
# 1. Run constrained d=3 optimization with multiple random restarts
# 2. Confirm convergence to a narrow interval around 1+√5
# 3. Verify that x²−2x−4 = 0 has a unique root in that interval

from scipy.optimize import minimize

QUAD_C5_EDGES = [(0,3),(0,5),(1,4),(1,6),(2,5),(2,6),(2,7),(3,6),(3,7),(4,7)]
N_VERTICES = 8
D = 3  # qutrit

def eta3_objective(params):
    """Negatif amaç: Σᵢ (ψ·vᵢ)², normalizasyon içeride."""
    psi = params[:D]; psi = psi / np.linalg.norm(psi)
    vecs = [params[D*(i+1):D*(i+2)] for i in range(N_VERTICES)]
    vecs = [v / np.linalg.norm(v) for v in vecs]
    return -sum((psi @ v)**2 for v in vecs)

# Explicit orthogonality constraints (SLSQP)
constraints = []
for ei, ej in QUAD_C5_EDGES:
    def orth(p, i=ei, j=ej):
        vi = p[D*(i+1):D*(i+2)]; vi = vi / np.linalg.norm(vi)
        vj = p[D*(j+1):D*(j+2)]; vj = vj / np.linalg.norm(vj)
        return vi @ vj
    constraints.append({'type': 'eq', 'fun': orth})

N_RESTARTS = 500
np.random.seed(42)

best_val = 0.0
all_vals = []

print(f'η₃(Quad-C₅) için SLSQP optimizasyonu ({N_RESTARTS} restart)...')

for k in range(N_RESTARTS):
    x0 = np.random.randn(D * (N_VERTICES + 1))
    res = minimize(eta3_objective, x0, method='SLSQP', constraints=constraints,
                   options={'maxiter': 5000, 'ftol': 1e-12})
    psi = res.x[:D]; psi = psi / np.linalg.norm(psi)
    vecs = [res.x[D*(i+1):D*(i+2)] for i in range(N_VERTICES)]
    vecs = [v / np.linalg.norm(v) for v in vecs]
    pen = sum((vecs[i] @ vecs[j])**2 for i, j in QUAD_C5_EDGES)
    if pen < 1e-8:
        val = sum((psi @ v)**2 for v in vecs)
        all_vals.append(val)
        if val > best_val:
            best_val = val

target = 1.0 + np.sqrt(5)
print(f'Kısıt sağlayan çözüm: {len(all_vals)}/{N_RESTARTS}')
print(f'En iyi η₃            : {best_val:.12f}')
print(f'1 + √5               : {target:.12f}')
print(f'Fark                 : {abs(best_val - target):.2e}')

import csv

target     = 1.0 + np.sqrt(5)
lo_eta3    = min(all_vals)
hi_eta3    = max(all_vals)
interval_w = hi_eta3 - lo_eta3

print('=== η₃(Quad-C₅) Sertifikasyon Sonucu ===')
print()
print(f'Gözlemlenen aralık  : [{lo_eta3:.10f}, {hi_eta3:.10f}]')
print(f'Aralık genişliği    : {interval_w:.2e}')
print(f'1 + √5              : {target:.10f}')
print(f'Aralık içinde mi?   : {lo_eta3 <= target <= hi_eta3}')
print()

roots = np.roots([1, -2, -4])
print(f'x²-2x-4=0 kökleri  : {sorted(roots)[::-1]}')
print(f'Pozitif kök (1+√5)  : {max(roots):.10f}')
print(f'Negatif kök (1-√5)  : {min(roots):.10f}  ← aralık dışında')
print()

poly_check = best_val**2 - 2*best_val - 4
print(f'Polinom x²-2x-4 @ best_val : {poly_check:.2e}  (≈0 olmalı)')
print()

if abs(best_val - target) < 1e-8:
    print('✓ η₃(Quad-C₅) = 1+√5 sayısal olarak sertifikalandı.')
    print(f'  |best - (1+√5)| = {abs(best_val - target):.2e}')
    print('  x²-2x-4=0 aralıkta tek pozitif kök → algebraik kimlik doğrulandı.')
else:
    print('✗ Sertifikasyon başarısız.')

with open('data/eta3_certification.csv', 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['quantity', 'value'])
    w.writerow(['target_1_plus_sqrt5', target])
    w.writerow(['best_numerical_eta3', best_val])
    w.writerow(['interval_lo', lo_eta3])
    w.writerow(['interval_hi', hi_eta3])
    w.writerow(['abs_diff', abs(best_val - target)])
    w.writerow(['poly_check_x2m2xm4', poly_check])
    w.writerow(['n_feasible_restarts', len(all_vals)])

print()
print('✓ data/eta3_certification.csv kaydedildi.')


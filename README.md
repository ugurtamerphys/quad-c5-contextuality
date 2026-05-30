# Quad-C5: Maximum Contextuality Gap on Eight Vertices

Code and data accompanying the paper:

> **The Quad-C5 Graph: Maximum Contextuality Gap on Eight Vertices**  
> U. Tamer, Ö. E. Müstecaplıoğlu, A. Dizdar, Z. Gedik  
> Submitted (2026)

## Repository Contents

| File | Description |
|------|-------------|
| `certification.py` | Exhaustive SDP search over all 11,117 connected 8-vertex graphs, Lovász theta computation, primal–dual certification, dimension classification |
| `noise_robustness.py` | Critical visibility under depolarizing noise for KCBS, Quad-C5, Wagner, and C7 |
| `dual_certificate.py` | Dual SDP certificate for the Quad-C5 Lovász theta value |
| `graph8.g6` | McKay graph database: all non-isomorphic simple graphs on 8 vertices (source: https://users.cecs.anu.edu.au/~bdm/data/) |
| `data/all_n8_results.csv` | Independence number and Lovász theta for all 11,117 connected 8-vertex graphs |
| `data/top50_certification.csv` | High-precision CLARABEL results for the top-50 graphs |
| `data/certified_intervals.csv` | SCS + CLARABEL interval bounds for the top-3 graphs |
| `data/eta3_certification.csv` | Dimension classification results (η₃ optimization, 300 restarts) |

## Requirements

```
pip install numpy networkx cvxpy scipy matplotlib tqdm mpmath
```

Tested with Python 3.10+. The SDP solver CLARABEL is installed automatically with CVXPY.

## Usage

Run `certification.py` to reproduce the full exhaustive search and all tables in the paper.  
Run `noise_robustness.py` to reproduce the noise robustness analysis and figure.  
Run `dual_certificate.py` for a standalone check of the Quad-C5 dual certificate.


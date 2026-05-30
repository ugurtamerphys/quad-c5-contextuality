"""
Dual SDP certificate for Quad-C5 Lovász theta.
Implements graphCertification.pdf Section 5 protocol.
Reports: theta_primal, y_dual, dual_gap, lambda_min(S).
"""

import numpy as np
import cvxpy as cp

QUAD_C5_EDGES = [(0,3),(0,5),(1,4),(1,6),(2,5),(2,6),(2,7),(3,6),(3,7),(4,7)]
N = 8


def lovasz_theta_with_dual_certificate(n, edges, solver="CLARABEL"):
    J = np.ones((n, n))
    X = cp.Variable((n, n), symmetric=True)

    # Build B matrices for each edge
    B_list = []
    for (i, j) in edges:
        B = np.zeros((n, n))
        B[i, j] = 0.5
        B[j, i] = 0.5
        B_list.append(B)

    constraints = [X >> 0]
    trace_constraint = (cp.trace(X) == 1)
    constraints.append(trace_constraint)

    edge_constraints = []
    for B in B_list:
        con = (cp.trace(B @ X) == 0)
        edge_constraints.append(con)
        constraints.append(con)

    objective = cp.Maximize(cp.trace(J @ X))
    problem = cp.Problem(objective, constraints)
    problem.solve(solver=solver, verbose=False)

    if problem.status not in ("optimal", "optimal_inaccurate"):
        print(f"Solver status: {problem.status}")
        return None

    Xval = X.value
    primal_value = np.trace(J @ Xval)

    # Extract dual variables (try both sign conventions)
    y_raw = trace_constraint.dual_value
    z_raw = np.array([con.dual_value for con in edge_constraints], dtype=float)

    best = None
    for sign in [+1, -1]:
        y = sign * y_raw
        z = sign * z_raw
        S = y * np.eye(n) - J.copy()
        for ze, B in zip(z, B_list):
            S += ze * B
        S_sym = (S + S.T) / 2
        eigs = np.linalg.eigvalsh(S_sym)
        min_eig = eigs[0]
        dual_gap = y - primal_value
        candidate = {
            "sign": sign,
            "y_dual": y,
            "primal": primal_value,
            "dual_gap": dual_gap,
            "lambda_min_S": min_eig,
            "S": S_sym,
        }
        if best is None or abs(dual_gap) < abs(best["dual_gap"]):
            best = candidate

    return best


if __name__ == "__main__":
    print("=== Quad-C5 Dual SDP Certificate ===\n")
    result = lovasz_theta_with_dual_certificate(N, QUAD_C5_EDGES, solver="CLARABEL")

    if result is None:
        print("Solver failed.")
    else:
        print(f"theta_primal          = {result['primal']:.10f}")
        print(f"y_dual                = {result['y_dual']:.10f}")
        print(f"dual_gap (y - primal) = {result['dual_gap']:.2e}")
        print(f"lambda_min(S)         = {result['lambda_min_S']:.2e}")
        print()
        if result["lambda_min_S"] >= -1e-6:
            print("CERTIFICATE VALID: S >= 0 (within numerical tolerance)")
        else:
            print("WARNING: S not PSD, certificate invalid")
        print()
        print("Reporting quantities for paper Table:")
        print(f"  vartheta_primal = {result['primal']:.8f}")
        print(f"  y_dual          = {result['y_dual']:.8f}")
        print(f"  r_d = y - Tr(JX)= {result['dual_gap']:.2e}")
        print(f"  lambda_min(S)   = {result['lambda_min_S']:.2e}")

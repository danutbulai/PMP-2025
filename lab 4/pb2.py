import numpy as np
from pgmpy.models import MarkovNetwork
from pgmpy.factors.discrete import DiscreteFactor
from pgmpy.inference import BeliefPropagation

np.random.seed(42)

H, W = 5, 5
orig = np.random.randint(0, 2, size=(H, W))

noisy = orig.copy()
num_pixels = H * W
num_noisy = max(1, num_pixels // 10)

indices = np.random.choice(num_pixels, size=num_noisy, replace=False)
for idx in indices:
    r = idx // W
    c = idx % W
    noisy[r, c] = 1 - noisy[r, c]

print("Original image:")
print(orig)
print("\nNoisy image:")
print(noisy)

mn = MarkovNetwork()

nodes = []
for i in range(H):
    for j in range(W):
        nodes.append(f"X_{i}_{j}")

mn.add_nodes_from(nodes)

edges = []
for i in range(H):
    for j in range(W):
        if i + 1 < H:
            edges.append((f"X_{i}_{j}", f"X_{i+1}_{j}"))
        if j + 1 < W:
            edges.append((f"X_{i}_{j}", f"X_{i}_{j+1}"))

mn.add_edges_from(edges)

lam = 2.0
factors = []

for i in range(H):
    for j in range(W):
        var = f"X_{i}_{j}"
        y_ij = noisy[i, j]
        vals = []
        for xi in [0, 1]:
            energy = lam * (xi - y_ij) ** 2
            vals.append(np.exp(-energy))
        phi_i = DiscreteFactor(variables=[var], cardinality=[2], values=vals)
        factors.append(phi_i)

pair_vals = np.zeros((2, 2))
for xi in [0, 1]:
    for xj in [0, 1]:
        energy = (xi - xj) ** 2
        pair_vals[xi, xj] = np.exp(-energy)

for (u, v) in edges:
    phi_uv = DiscreteFactor(variables=[u, v], cardinality=[2, 2], values=pair_vals)
    factors.append(phi_uv)

mn.add_factors(*factors)

mn.check_model()

bp = BeliefPropagation(mn)

all_vars = [f"X_{i}_{j}" for i in range(H) for j in range(W)]

map_result = bp.map_query(variables=all_vars)

denoised = np.zeros((H, W), dtype=int)
for i in range(H):
    for j in range(W):
        denoised[i, j] = map_result[f"X_{i}_{j}"]

print("\nDenoised image (MAP estimate):")
print(denoised)

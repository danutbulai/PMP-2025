import numpy as np
import matplotlib.pyplot as plt

# 1)
x1 = np.random.poisson(1, 1000)
x2 = np.random.poisson(2, 1000)
x3 = np.random.poisson(5, 1000)
x4 = np.random.poisson(10, 1000)

# 2)
L = [1, 2, 5, 10]
rand_eq = np.array([np.random.poisson(np.random.choice(L)) for _ in range(1000)])

# a)
plt.figure(figsize=(10,6))
plt.hist(x1, bins=range(0,35), alpha=0.5, label="lambda=1")
plt.hist(x2, bins=range(0,35), alpha=0.5, label="lambda=2")
plt.hist(x3, bins=range(0,35), alpha=0.5, label="lambda=5")
plt.hist(x4, bins=range(0,35), alpha=0.5, label="lambda=10")
plt.hist(rand_eq, bins=range(0,35), alpha=0.5, label="random lambda")
plt.legend()
plt.title("a) Distributii Poisson (fixe + random)")
plt.xlabel("valoare")
plt.ylabel("frecventa")
plt.show()

# b)
print("b) Distributiile fixe sunt stranse in jurul valorii lambda lor.")
print("Cea random e mai lata, combina toate formele.")
print("Cand lambda variaza, distributia devine mai raspandita, incertitudine mai mare.")

# c)
w = [0.1, 0.1, 0.6, 0.2]  # lambda=5 mai probabil
rand_w = np.array([np.random.poisson(np.random.choice(L, p=w)) for _ in range(1000)])

plt.figure(figsize=(10,6))
plt.hist(rand_eq, bins=range(0,35), alpha=0.5, label="random egal")
plt.hist(rand_w, bins=range(0,35), alpha=0.5, label="random spre lambda=5")
plt.legend()
plt.title("c) lambda=5 mai probabil")
plt.xlabel("valoare")
plt.ylabel("frecventa")
plt.show()

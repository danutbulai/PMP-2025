import numpy as np
import pandas as pd
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt
from multiprocessing import freeze_support

def fit_mixture_poly(x, y, K, draws=500, tune=500, target_accept=0.9):
    x2 = x**2

    with pm.Model() as model:
        w = pm.Dirichlet("w", a=np.ones(K))

        alpha = pm.Normal("alpha", 0, 10, shape=K)
        beta = pm.Normal("beta", 0, 10, shape=K)
        gamma = pm.Normal("gamma", 0, 10, shape=K)
        sigma = pm.HalfNormal("sigma", 10, shape=K)

        mu = alpha[:, None] + beta[:, None] * x + gamma[:, None] * x2
        comp = pm.Normal.dist(mu=mu.T, sigma=sigma)

        pm.Mixture("y_obs", w=w, comp_dists=comp, observed=y)

        idata = pm.sample(draws=draws, tune=tune, chains=1, cores=1, target_accept=target_accept, random_seed=42)

    return idata

def summarize_params(idata, K):
    w_m = idata.posterior["w"].mean(("chain", "draw")).values
    a_m = idata.posterior["alpha"].mean(("chain", "draw")).values
    b_m = idata.posterior["beta"].mean(("chain", "draw")).values
    g_m = idata.posterior["gamma"].mean(("chain", "draw")).values
    s_m = idata.posterior["sigma"].mean(("chain", "draw")).values
    return w_m, a_m, b_m, g_m, s_m

def plot_mixture_fit(x, y, idata, K, title):
    x_plot = np.linspace(x.min(), x.max(), 300)
    x2_plot = x_plot**2

    w_m, a_m, b_m, g_m, _ = summarize_params(idata, K)

    plt.figure(figsize=(8, 5))
    plt.scatter(x, y, s=10, alpha=0.6)

    for k in range(K):
        yk = a_m[k] + b_m[k] * x_plot + g_m[k] * x2_plot
        plt.plot(x_plot, yk, label=f"comp {k+1}, w={w_m[k]:.2f}")

    plt.xlabel("Ore exercitii (standardizat)")
    plt.ylabel("Colesterol (standardizat)")
    plt.title(title)
    plt.legend()
    plt.show()

def main():
    df = pd.read_csv("date_colesterol.csv")

    x_raw = df["Ore_Exercitii"].values
    y_raw = df["Colesterol"].values

    x = (x_raw - x_raw.mean()) / x_raw.std()
    y = (y_raw - y_raw.mean()) / y_raw.std()

    models = {}
    waic_table = {}
    loo_table = {}

    for K in [3, 4, 5]:
        idata = fit_mixture_poly(x, y, K)
        models[K] = idata

        w_m, a_m, b_m, g_m, s_m = summarize_params(idata, K)

        print(f"\nK={K} — estimari medii (posterior mean)")
        for k in range(K):
            print(f"  comp {k+1}: w={w_m[k]:.3f}, alpha={a_m[k]:.3f}, beta={b_m[k]:.3f}, gamma={g_m[k]:.3f}, sigma={s_m[k]:.3f}")

        waic = az.waic(idata)
        loo = az.loo(idata)

        waic_table[K] = waic
        loo_table[K] = loo

        plot_mixture_fit(x, y, idata, K, f"Mixture polynomial model (K={K})")

    cmp_waic = az.compare({f"K={k}": models[k] for k in models}, ic="waic")
    cmp_loo = az.compare({f"K={k}": models[k] for k in models}, ic="loo")

    print("\nWAIC comparison:")
    print(cmp_waic)

    print("\nLOO comparison:")
    print(cmp_loo)

    best_waic = cmp_waic.index[0]
    best_loo = cmp_loo.index[0]

    print("\n1) **Estimarea ponderilor si coeficientilor**:")
    print("   **Am folosit un model mixt (mixture) cu K componente, unde w ~ Dirichlet, iar fiecare componenta are un model polinomial de ordin 2:**")
    print("   **mu_k(x) = alpha_k + beta_k * x + gamma_k * x^2, iar observatiile sunt modelate cu o mixtura de Gaussiene.**")
    print("   **Prin sampling (NUTS) am estimat w, alpha, beta, gamma si sigma pentru fiecare K in {3,4,5}.**")

    print("\n2) **Cate subpopulatii descriu cel mai bine datele?**:")
    print("   **Am comparat modelele folosind WAIC si LOO. Modelul preferat este cel cu scor mai mic (mai bun).**")
    print(f"   **Best dupa WAIC: {best_waic}**")
    print(f"   **Best dupa LOO : {best_loo}**")
    print("   **Justificare: WAIC/LOO penalizeaza modelele prea complexe, deci alegem K-ul care explica bine datele fara overfitting.**")

    az.plot_compare(cmp_waic)
    plt.title("Model comparison (WAIC)")
    plt.show()

    az.plot_compare(cmp_loo)
    plt.title("Model comparison (LOO)")
    plt.show()

if __name__ == "__main__":
    freeze_support()
    main()

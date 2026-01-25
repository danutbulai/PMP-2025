import numpy as np
import pandas as pd
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt
from multiprocessing import freeze_support

def fit_poly_model(x, y, order, beta_sd, draws=500, tune=500, target_accept=0.9):
    X = np.vstack([x**i for i in range(1, order + 1)])

    with pm.Model() as m:
        alpha = pm.Normal("alpha", 0, 1)
        if isinstance(beta_sd, np.ndarray):
            beta = pm.Normal("beta", 0, beta_sd, shape=order)
        else:
            beta = pm.Normal("beta", 0, beta_sd, shape=order)
        eps = pm.HalfNormal("eps", 1)

        mu = alpha + pm.math.dot(beta, X)
        pm.Normal("y_obs", mu=mu, sigma=eps, observed=y)

        idata = pm.sample(draws=draws, tune=tune, chains=1, cores=1, target_accept=target_accept, random_seed=42)

    alpha_m = idata.posterior["alpha"].mean().item()
    beta_m = idata.posterior["beta"].mean(("chain", "draw")).values
    return idata, alpha_m, beta_m

def poly_predict(alpha_m, beta_m, x_plot, order):
    X_plot = np.vstack([x_plot**i for i in range(1, order + 1)])
    return alpha_m + beta_m @ X_plot

def plot_fit(x, y, x_plot, y_plot, title):
    plt.figure(figsize=(8, 5))
    plt.scatter(x, y, s=10)
    plt.plot(x_plot, y_plot)
    plt.title(title)
    plt.xlabel("x (standardizat)")
    plt.ylabel("y (standardizat)")
    plt.show()

def main():
    df = pd.read_csv("date.csv", sep=r"\s+", header=None)
    x_raw = df.iloc[:, 0].values
    y_raw = df.iloc[:, 1].values

    x = (x_raw - x_raw.mean()) / x_raw.std()
    y = (y_raw - y_raw.mean()) / y_raw.std()

    order = 5
    x_plot = np.linspace(x.min(), x.max(), 300)

    idata_5_sd10, a10, b10 = fit_poly_model(x, y, order=order, beta_sd=10, draws=500, tune=500)
    y_plot_5_sd10 = poly_predict(a10, b10, x_plot, order)
    plot_fit(x, y, x_plot, y_plot_5_sd10, "1a) order=5, beta~N(0,10)")

    idata_5_sd100, a100, b100 = fit_poly_model(x, y, order=order, beta_sd=100, draws=500, tune=500)
    y_plot_5_sd100 = poly_predict(a100, b100, x_plot, order)
    plot_fit(x, y, x_plot, y_plot_5_sd100, "1b) order=5, beta~N(0,100)")

    beta_sd_vec = np.array([10, 0.1, 0.1, 0.1, 0.1])
    idata_5_vec, av, bv = fit_poly_model(x, y, order=order, beta_sd=beta_sd_vec, draws=500, tune=500)
    y_plot_5_vec = poly_predict(av, bv, x_plot, order)
    plot_fit(x, y, x_plot, y_plot_5_vec, "1b) order=5, beta~N(0,[10,0.1,0.1,0.1,0.1])")

    print("1b) **Diferente**:")
    print("   - sd=10: regularizare moderata, curba mai stabila.")
    print("   - sd=100: prior foarte slab (aproape plat), curba poate oscila mai mult (overfitting).")
    print("   - sd=[10,0.1,0.1,0.1,0.1]: termeni de ordin mare sunt puternic regularizati, curba devine mai neteda.\n")

    np.random.seed(42)
    n = 500
    idx = np.random.choice(len(x), size=n, replace=True)
    x500 = x[idx]
    y500 = y[idx]

    idata_5_500, a5_500, b5_500 = fit_poly_model(x500, y500, order=order, beta_sd=10, draws=500, tune=500)
    y_plot_5_500 = poly_predict(a5_500, b5_500, x_plot, order)
    plot_fit(x500, y500, x_plot, y_plot_5_500, "2) order=5, N=500 (bootstrap), beta~N(0,10)")

    order1 = 1
    idata_l, al, bl = fit_poly_model(x, y, order=order1, beta_sd=10, draws=700, tune=700)
    y_plot_l = poly_predict(al, bl, x_plot, order1)

    order2 = 2
    idata_q, aq, bq = fit_poly_model(x, y, order=order2, beta_sd=10, draws=700, tune=700)
    y_plot_q = poly_predict(aq, bq, x_plot, order2)

    order3 = 3
    idata_c, ac, bc = fit_poly_model(x, y, order=order3, beta_sd=10, draws=700, tune=700)
    y_plot_c = poly_predict(ac, bc, x_plot, order3)

    plt.figure(figsize=(8, 5))
    plt.scatter(x, y, s=10)
    plt.plot(x_plot, y_plot_l, label="linear (order=1)")
    plt.plot(x_plot, y_plot_q, label="quadratic (order=2)")
    plt.plot(x_plot, y_plot_c, label="cubic (order=3)")
    plt.title("3) Comparatie curbe: linear vs quadratic vs cubic")
    plt.xlabel("x (standardizat)")
    plt.ylabel("y (standardizat)")
    plt.legend()
    plt.show()

    waic_l = az.waic(idata_l)
    waic_q = az.waic(idata_q)
    waic_c = az.waic(idata_c)

    loo_l = az.loo(idata_l)
    loo_q = az.loo(idata_q)
    loo_c = az.loo(idata_c)

    cmp_waic = az.compare({"linear": idata_l, "quadratic": idata_q, "cubic": idata_c}, ic="waic")
    cmp_loo = az.compare({"linear": idata_l, "quadratic": idata_q, "cubic": idata_c}, ic="loo")

    print("3) **WAIC**:")
    print(cmp_waic, "\n")
    print("3) **LOO**:")
    print(cmp_loo, "\n")

    az.plot_compare(cmp_waic)
    plt.title("3) Model comparison (WAIC)")
    plt.show()

    az.plot_compare(cmp_loo)
    plt.title("3) Model comparison (LOO)")
    plt.show()

if __name__ == "__main__":
    freeze_support()
    main()

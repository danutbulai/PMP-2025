import numpy as np
import pandas as pd
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt
from multiprocessing import freeze_support

def main():
    data = pd.read_csv("date_promovare_examen.csv")

    X_study = data["Ore_Studiu"].values
    X_sleep = data["Ore_Somn"].values
    y = data["Promovare"].values

    unique, counts = np.unique(y, return_counts=True)
    dist = dict(zip(unique, counts))
    p0 = dist.get(0, 0) / len(y)
    p1 = dist.get(1, 0) / len(y)
    balanced = abs(p0 - p1) <= 0.10

    print(f"a) **Distributie clase**: {dist} (total={len(y)})")
    print(f"   **Proportii**: nepromovat={p0:.3f}, promovat={p1:.3f}")
    print(f"   **Concluzie**: datele sunt {'balansate' if balanced else 'debalansate'}.\n")

    X_study_std = (X_study - X_study.mean()) / X_study.std()
    X_sleep_std = (X_sleep - X_sleep.mean()) / X_sleep.std()

    with pm.Model() as model_logistic:
        alpha = pm.Normal("alpha", mu=0, sigma=5)
        beta_study = pm.Normal("beta_study", mu=0, sigma=5)
        beta_sleep = pm.Normal("beta_sleep", mu=0, sigma=5)

        logits = alpha + beta_study * X_study_std + beta_sleep * X_sleep_std
        p = pm.Deterministic("p", pm.math.sigmoid(logits))

        pm.Bernoulli("y_obs", p=p, observed=y)

        trace = pm.sample(draws=1000, tune=1000, target_accept=0.9, random_seed=42, chains=1, cores=1)

    print("a) **Modelul Bayesian de regresie logistica a fost antrenat (sampling finalizat).**\n")

    alpha_m = trace.posterior["alpha"].mean().item()
    beta_study_m = trace.posterior["beta_study"].mean().item()
    beta_sleep_m = trace.posterior["beta_sleep"].mean().item()

    print("b) **Granita de decizie medie (pe date standardizate)**:")
    print(f"   alpha_mean = {alpha_m:.4f}")
    print(f"   beta_study_mean = {beta_study_m:.4f}")
    print(f"   beta_sleep_mean = {beta_sleep_m:.4f}")
    print("   Formula: alpha + beta_study*x_study + beta_sleep*x_sleep = 0\n")

    logits_mean = alpha_m + beta_study_m * X_study_std + beta_sleep_m * X_sleep_std
    p_mean = 1 / (1 + np.exp(-logits_mean))
    y_pred = (p_mean >= 0.5).astype(int)
    acc = (y_pred == y).mean()

    print(f"b) **Acuratete (predictii cu media posteriorului, prag 0.5)**: {acc:.3f}\n")

    x_vals = np.linspace(X_study_std.min(), X_study_std.max(), 200)
    y_vals = -(alpha_m + beta_study_m * x_vals) / beta_sleep_m

    plt.figure(figsize=(8, 6))
    plt.scatter(X_study_std[y == 0], X_sleep_std[y == 0], alpha=0.6, label="Nepromovat (0)")
    plt.scatter(X_study_std[y == 1], X_sleep_std[y == 1], alpha=0.6, label="Promovat (1)")
    plt.plot(x_vals, y_vals, linewidth=2, label="Granita decizie medie")
    plt.xlabel("Ore_Studiu (standardizat)")
    plt.ylabel("Ore_Somn (standardizat)")
    plt.title("Regresie logistica Bayesiana — granita de decizie (medie posterior)")
    plt.legend()
    plt.show()

    summary = az.summary(trace, var_names=["beta_study", "beta_sleep"])
    b_study_mean = summary.loc["beta_study", "mean"]
    b_sleep_mean = summary.loc["beta_sleep", "mean"]

    print("c) **Comparatie influenta (coeficienti pe date standardizate)**:")
    print(summary[["mean", "sd", "hdi_3%", "hdi_97%"]])

    more = "orele de studiu" if abs(b_study_mean) > abs(b_sleep_mean) else "orele de somn"
    print(f"\nc) **Concluzie**: influenteaza mai mult **{more}**.\n")

    az.plot_posterior(trace, var_names=["alpha", "beta_study", "beta_sleep"])
    plt.show()

if __name__ == "__main__":
    freeze_support()
    main()

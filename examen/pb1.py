import numpy as np
import pandas as pd
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt
from multiprocessing import freeze_support

def zscore(a):
    a = np.asarray(a, dtype=float)
    return (a - a.mean()) / a.std()

def build_design_matrix(df, include_temp2=False):
    x_temp = zscore(df["temp_c"].values)
    x_hum = zscore(df["humidity"].values)
    x_wind = zscore(df["wind_kph"].values)
    x_holiday = df["is_holiday"].astype(int).values
    season = df["season"].astype(str)
    season_d = pd.get_dummies(season, drop_first=True)
    season_cols = list(season_d.columns)
    cols = ["temp_c", "humidity", "wind_kph", "is_holiday"] + [f"season_{c}" for c in season_cols]
    X_parts = [x_temp, x_hum, x_wind, x_holiday.astype(float)]
    for c in season_cols:
        X_parts.append(season_d[c].values.astype(float))

    if include_temp2:
        x_temp2 = x_temp ** 2
        X_parts.append(x_temp2)
        cols.append("temp_c2")

    X = np.column_stack(X_parts)
    return X, cols

def fit_bayes_linear_regression(X, y, draws=1500, tune=1000, chains=2, cores=1, target_accept=0.9, seed=42):
    with pm.Model() as m:
        alpha = pm.Normal("alpha", 0, 1)
        beta = pm.Normal("beta", 0, 1, shape=X.shape[1])
        sigma = pm.HalfNormal("sigma", 1)

        mu = alpha + pm.math.dot(X, beta)
        pm.Normal("y_obs", mu=mu, sigma=sigma, observed=y)

        idata = pm.sample(
            draws=draws,
            tune=tune,
            chains=chains,
            cores=cores,
            target_accept=target_accept,
            random_seed=seed,
            progressbar=True,
        )
    return idata

def fit_bayes_logistic_regression(X, y_bin, draws=1500, tune=1000, chains=2, cores=1, target_accept=0.9, seed=42):
    with pm.Model() as m:
        alpha = pm.Normal("alpha", 0, 1)
        beta = pm.Normal("beta", 0, 1, shape=X.shape[1])

        logits = alpha + pm.math.dot(X, beta)
        p = pm.Deterministic("p", pm.math.sigmoid(logits))

        pm.Bernoulli("y_obs", p=p, observed=y_bin)

        idata = pm.sample(
            draws=draws,
            tune=tune,
            chains=chains,
            cores=cores,
            target_accept=target_accept,
            random_seed=seed,
            progressbar=True,
        )
    return idata

def coef_influence(idata, colnames):
    beta_samples = idata.posterior["beta"].stack(s=("chain", "draw")).values
    abs_mean = np.mean(np.abs(beta_samples), axis=1)
    order = np.argsort(-abs_mean)
    return [(colnames[i], abs_mean[i]) for i in order]

def ppc_plot_vs_temp(df, idata, X, colnames, title):
    x_temp = zscore(df["temp_c"].values)
    idx = np.argsort(x_temp)
    x_sorted = x_temp[idx]
    y_sorted = zscore(df["rentals"].values)[idx]

    with pm.Model() as m_tmp:
        alpha = pm.Normal("alpha", 0, 1)
        beta = pm.Normal("beta", 0, 1, shape=X.shape[1])
        sigma = pm.HalfNormal("sigma", 1)
        mu = alpha + pm.math.dot(X, beta)
        y_rep = pm.Normal("y_rep", mu=mu, sigma=sigma, shape=X.shape[0])

        idata_ppc = pm.sample_posterior_predictive(
            idata,
            var_names=["y_rep"],
            random_seed=42,
            progressbar=True
        )

def main():
    df = pd.read_csv("bike_daily.csv")

    print("1) Pairwise scatterplots")
    plt.figure(figsize=(6, 4))
    plt.scatter(df["temp_c"], df["rentals"], s=10, alpha=0.6)
    plt.xlabel("temp_c")
    plt.ylabel("rentals")
    plt.title("rentals vs temp_c")
    plt.show()

    plt.figure(figsize=(6, 4))
    plt.scatter(df["humidity"], df["rentals"], s=10, alpha=0.6)
    plt.xlabel("humidity")
    plt.ylabel("rentals")
    plt.title("rentals vs humidity")
    plt.show()

    plt.figure(figsize=(6, 4))
    plt.scatter(df["wind_kph"], df["rentals"], s=10, alpha=0.6)
    plt.xlabel("wind_kph")
    plt.ylabel("rentals")
    plt.title("rentals vs wind_kph")
    plt.show()

    print("2) Standardizare: temp_c, humidity, wind_kph si target rentals")
    y_std = zscore(df["rentals"].values)

    X_lin, cols_lin = build_design_matrix(df, include_temp2=False)
    X_poly, cols_poly = build_design_matrix(df, include_temp2=True)

    print("2b) Bayesian linear regression (fara temp_c2)")
    idata_lin = fit_bayes_linear_regression(X_lin, y_std)

    print("2c) Polynomial regression (cu temp_c2)")
    idata_poly = fit_bayes_linear_regression(X_poly, y_std)

    print("3) Posterior summaries (linear):")
    print(az.summary(idata_lin, var_names=["alpha", "beta", "sigma"]))

    print("3) Posterior summaries (poly):")
    print(az.summary(idata_poly, var_names=["alpha", "beta", "sigma"]))

    infl_lin = coef_influence(idata_lin, cols_lin)
    infl_poly = coef_influence(idata_poly, cols_poly)

    print("\n3) Variabile influente (linear, ordonate):")
    for name, val in infl_lin[:8]:
        print(f"   {name}: {val:.3f}")

    print("\n3) Variabile influente (poly, ordonate):")
    for name, val in infl_poly[:9]:
        print(f"   {name}: {val:.3f}")

    print("\n4a) (WAIC si LOO)")
    cmp_waic = az.compare({"linear": idata_lin, "poly": idata_poly}, ic="waic")
    cmp_loo = az.compare({"linear": idata_lin, "poly": idata_poly}, ic="loo")
    print("\nWAIC:")
    print(cmp_waic)
    print("\nLOO:")
    print(cmp_loo)

    az.plot_compare(cmp_waic)
    plt.title("Compare models (WAIC)")
    plt.show()

    az.plot_compare(cmp_loo)
    plt.title("Compare models (LOO)")
    plt.show()

    print("\n4b) PPC:")
    ppc_plot_vs_temp(df, idata_lin, X_lin, cols_lin, "PPC (linear) vs temp_c")
    ppc_plot_vs_temp(df, idata_poly, X_poly, cols_poly, "PPC (poly) vs temp_c")

if __name__ == "__main__":
        freeze_support()
        main()
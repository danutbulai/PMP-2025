import numpy as np
import pymc as pm
import arviz as az

publicity = np.array([
    1.5, 2.0, 2.3, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0,
    6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0, 10.5, 11.0
])

sales = np.array([
    5.2, 6.8, 7.5, 8.0, 9.0, 10.2, 11.5, 12.0, 13.5, 14.0,
    15.0, 15.5, 16.2, 17.0, 18.0, 18.5, 19.5, 20.0, 21.0, 22.0
])

new_publicity = np.array([2.0, 6.0, 10.0])


def main():
    with pm.Model() as model:
        alpha = pm.Normal("alpha", mu=0, sigma=10)
        beta = pm.Normal("beta", mu=0, sigma=10)
        sigma = pm.HalfNormal("sigma", sigma=5)

        mu = alpha + beta * publicity
        pm.Normal("sales", mu=mu, sigma=sigma, observed=sales)

        idata = pm.sample(
            2000,
            tune=1000,
            target_accept=0.9,
            random_seed=1,
            cores=1
        )

    print("\nRezumat coeficienti:")
    print(az.summary(idata, var_names=["alpha", "beta"]))

    coef_hdi = az.hdi(idata, var_names=["alpha", "beta"], hdi_prob=0.94)
    print("\nHDI 94% pentru coeficienti:")
    print(coef_hdi)

    posterior = idata.posterior
    alpha_samples = posterior["alpha"].values
    beta_samples = posterior["beta"].values
    sigma_samples = posterior["sigma"].values

    mu_pred = alpha_samples[..., None] + beta_samples[..., None] * new_publicity[None, None, :]

    rng = np.random.default_rng(1)
    y_pred = rng.normal(mu_pred, sigma_samples[..., None])

    pred_int = np.percentile(y_pred, [5, 95], axis=(0, 1))

    print("\nIntervale predictive (5%, 95%) pentru noile valori de publicity:")
    for p, lo, hi in zip(new_publicity, pred_int[0], pred_int[1]):
        print(f"publicity = {p:4.1f} -> sales ~ {lo:5.2f} .. {hi:5.2f}")


if __name__ == "__main__":
    main()

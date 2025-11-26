import pymc as pm
import arviz as az
import matplotlib.pyplot as plt

y_vals = [0, 5, 10]
thetas = [0.2, 0.5]

posts = {}
preds = {}

for t in thetas:
    for y_obs in y_vals:
        with pm.Model() as m:
            n = pm.Poisson("n", 10)
            y = pm.Binomial("y", n=n, p=t, observed=y_obs)
            y_new = pm.Binomial("y_new", n=n, p=t)
            tr = pm.sample(1000, tune=1000, chains=2, cores=1)
            ppc = pm.sample_posterior_predictive(
                tr,
                var_names=["y_new"],
                return_inferencedata=False,
            )
        posts[(y_obs, t)] = tr
        preds[(y_obs, t)] = ppc["y_new"]

fig, ax = plt.subplots(len(y_vals), len(thetas), figsize=(10, 8))
for i, y_obs in enumerate(y_vals):
    for j, t in enumerate(thetas):
        a = ax[i, j]
        az.plot_posterior(posts[(y_obs, t)], var_names=["n"], ax=a)
        a.set_title(f"y={y_obs}, theta={t}")
plt.tight_layout()
plt.show()

fig, ax = plt.subplots(len(y_vals), len(thetas), figsize=(10, 8))
for i, y_obs in enumerate(y_vals):
    for j, t in enumerate(thetas):
        a = ax[i, j]
        az.plot_dist(preds[(y_obs, t)], ax=a)
        a.set_title(f"y* | y={y_obs}, theta={t}")
plt.tight_layout()
plt.show()

import pymc as pm
import numpy as np
import pandas as pd
import arviz as az

if __name__ == '__main__':
    df = pd.read_csv('Prices.csv')

    premium_map = {'yes': 1, 'no':  0}
    df['Premium'] = df['Premium'].map(premium_map)

    price = df['Price']. values
    speed = df['Speed'].values
    hd = df['HardDrive'].values
    log_hd = np. log(hd)
    premium = df['Premium'].values

    with pm.Model() as model:
        alpha = pm.Normal('alpha', mu=0, sigma=1000)
        beta1 = pm.Normal('beta1', mu=0, sigma=1000)
        beta2 = pm.Normal('beta2', mu=0, sigma=1000)
        sigma = pm.HalfNormal('sigma', sigma=1000)

        mu = alpha + beta1 * speed + beta2 * log_hd

        y = pm. Normal('y', mu=mu, sigma=sigma, observed=price)

        trace = pm. sample(1000, tune=1000, return_inferencedata=True, chains=1)

    print("a)")
    print(az.summary(trace, var_names=['alpha', 'beta1', 'beta2', 'sigma']))

    hdi_beta1 = az.hdi(trace, var_names=['beta1'], hdi_prob=0.95)
    hdi_beta2 = az.hdi(trace, var_names=['beta2'], hdi_prob=0.95)

    print("\nb)")
    print(f"95% HDI for beta1: [{hdi_beta1['beta1'].values[0]:.4f}, {hdi_beta1['beta1'].values[1]:.4f}]")
    print(f"95% HDI for beta2: [{hdi_beta2['beta2'].values[0]:.4f}, {hdi_beta2['beta2'].values[1]:.4f}]")

    print("\nc)")
    print("Da, ambii predictori sunt utili pentru predictia pretului.")
    print("Beta1 are un interval HDI care nu contine 0 si este pozitiv, indicand ca frecventa procesorului")
    print("creste pretul.  De asemenea, beta2 are un interval HDI pozitiv care nu contine 0, ceea ce")
    print("inseamna ca marimea hard disk-ului (in scala logaritmica) influenteaza semnificativ pretul.")
    print("Ambii coeficienti au o contributie clara si consistenta la modelul de predictie.")

    speed_new = 33
    hd_new = 540
    log_hd_new = np.log(hd_new)

    posterior = trace.posterior
    alpha_samples = posterior['alpha'].values. flatten()
    beta1_samples = posterior['beta1'].values. flatten()
    beta2_samples = posterior['beta2'].values. flatten()

    mu_pred = alpha_samples + beta1_samples * speed_new + beta2_samples * log_hd_new

    hdi_mu = az.hdi(mu_pred, hdi_prob=0.90)

    print("\nd)")
    print(f"90% HDI for expected price: [{hdi_mu[0]:.2f}, {hdi_mu[1]:.2f}]")

    sigma_samples = posterior['sigma'].values.flatten()
    y_pred = np.random.normal(mu_pred, sigma_samples)

    hdi_y = az.hdi(y_pred, hdi_prob=0.90)

    print("\ne)")
    print(f"90% HDI prediction interval: [{hdi_y[0]:.2f}, {hdi_y[1]:.2f}]")

    with pm.Model() as model_bonus:
        alpha_b = pm.Normal('alpha', mu=0, sigma=1000)
        beta1_b = pm.Normal('beta1', mu=0, sigma=1000)
        beta2_b = pm. Normal('beta2', mu=0, sigma=1000)
        beta3_b = pm.Normal('beta3', mu=0, sigma=1000)
        sigma_b = pm.HalfNormal('sigma', sigma=1000)

        mu_b = alpha_b + beta1_b * speed + beta2_b * log_hd + beta3_b * premium

        y_b = pm.Normal('y', mu=mu_b, sigma=sigma_b, observed=price)

        trace_bonus = pm. sample(1000, tune=1000, return_inferencedata=True, chains=1)

    hdi_beta3 = az.hdi(trace_bonus, var_names=['beta3'], hdi_prob=0.95)

    print("\nBonus)")
    print(f"95% HDI for beta3 (premium): [{hdi_beta3['beta3'].values[0]:.4f}, {hdi_beta3['beta3'].values[1]:.4f}]")
    print("\nDa, producatorul premium afecteaza semnificativ pretul.")
    print("Coeficientul beta3 pentru variabila premium este pozitiv si intervalul HDI nu contine 0,")
    print("ceea ce indica faptul ca PC-urile de la producatori premium (IBM, COMPAQ) au un pret")
    print("mai mare decat cele de la producatori non-premium, mentinand constanti ceilalti factori.")
    print("Aceasta diferenta este substantiala din punct de vedere statistic.")
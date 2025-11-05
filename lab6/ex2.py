import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

# Datele de intrare (exemplu sintetic)
# TODO: Înlocuiește cu datele corecte din contextul cerinței, dacă este cazul.
k = 180    # Numărul total de apeluri observate (din cerință)
T = 10     # Intervalul de timp în ore (din cerință)

# Parametrii distribuției prior normală trunchiată la zero
# TODO: Înlocuiește priorul cu forma cerută de cerință.
# Prior conjugat: Gamma(a0, b0) in parametrizare shape–rate
a0, b0 = 1, 1
prior_pdf = stats.gamma.pdf(
    lambda_values := np.linspace(0.001, 30, 1000),
    a=a0,
    scale=1.0 / b0
)

# Un prior prost ales duce la rezultate eronate
# mu_prior = 0      # Presupunem că rata medie a priori este 0 apeluri/oră
# sigma_prior = 1    # Presupunem o deviație standard de 1 apel/oră

# Definim gama de valori pentru λ (rata medie de apeluri pe oră)
# (mutată mai sus pentru a fi folosită și la priorul Gamma)
# lambda_values = np.linspace(0.001, 30, 1000)  # Evităm zero pentru stabilitate numerică

# Funcția de verosimilitate Poisson
likelihood = stats.poisson.pmf(k, mu=T * lambda_values)

# Calculăm distribuția a posteriori ne-normalizată
# TODO: Înlocuiește calculele numerice cu forma de posterior cerută de cerință.
# Posterior conjugat analitic: Gamma(a0 + k, b0 + T) – densitatea e deja normalizată
unnormalized_posterior = stats.gamma.pdf(
    lambda_values,
    a=a0 + k,
    scale=1.0 / (b0 + T)
)

# Normalizăm distribuția a posteriori
posterior_pdf = unnormalized_posterior  # deja normalizată prin pdf-ul Gamma

# Calculăm media a posteriori a lui λ
mean_lambda = np.trapz(lambda_values * posterior_pdf, lambda_values)

# Calculăm modulul a posteriori al lui λ (valoarea λ unde posteriorul atinge maximul)
mode_index = np.argmax(posterior_pdf)
mode_lambda = lambda_values[mode_index]

# Calculăm un interval de 94% (aproximare de tip coadă egală pe grilă)
# TODO: Calculează un interval conform metodei cerute de cerință.
# (Lăsăm "cozi egale" pe grilă; pentru HDI exact poți folosi ArviZ sau sliding window.)
cumulative = np.cumsum(posterior_pdf) * (lambda_values[1] - lambda_values[0])
lower_idx = np.where(cumulative >= 0.03)[0][0]
upper_idx = np.where(cumulative <= 0.97)[0][-1]
lower_bound = lambda_values[lower_idx]
upper_bound = lambda_values[upper_idx]

print(f"Media a posteriori a lui λ: {mean_lambda:.4f}")
print(f"Modul a posteriori al lui λ: {mode_lambda:.4f}")
print(f"Intervalul HDI de 94% pentru λ: [{lower_bound:.4f}, {upper_bound:.4f}]\n")

# Vizualizarea distribuției a posteriori
# TODO: Actualizează titluri/legende dacă schimbi modelarea.
plt.figure(figsize=(10, 6))
plt.plot(lambda_values, posterior_pdf, label='Distribuția a posteriori a lui λ (Gamma)')
plt.axvline(mean_lambda, color='r', linestyle='--', label=f'Media = {mean_lambda:.2f}')
plt.axvline(mode_lambda, color='g', linestyle='--', label=f'Modul = {mode_lambda:.2f}')
plt.fill_between(lambda_values, posterior_pdf, where=(lambda_values >= lower_bound) & (lambda_values <= upper_bound), color='gray', alpha=0.5, label='Interval 94% (aprox.)')
plt.title('Distribuția a posteriori (conjugată, prior Gamma)')
plt.xlabel('λ (rata medie de apeluri pe oră)')
plt.ylabel('Densitatea de probabilitate')
plt.legend()
plt.grid(True)
plt.show()

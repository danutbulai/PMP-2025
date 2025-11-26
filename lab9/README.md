# Lab 8 – Ex. 1

Model:
- n ~ Poisson(10) = numarul total de clienti intr-o zi
- Y | n, theta ~ Binomial(n, theta) = numarul de cumparatori
- theta este cunoscut
- Au fost rulate toate combinarile: Y apartine {0, 5, 10}, theta apartine {0.2, 0.5}

## a) Posterior pentru n
Am folosit pm.sample si az.plot_posterior pentru a obtine distributia a posteriori a lui n. Toate cele 6 cazuri sunt afisate in aceeasi fereastra sub forma de 3×2 subplot-uri.

## b) Efectul lui Y si theta asupra posteriorului
- Cand Y creste, distributia a posteriori pentru n se deplaseaza spre valori mai mari, deoarece sunt necesari mai multi clienti pentru a explica un numar mai mare de cumparatori.
- Pentru acelasi Y, un theta mai mare inseamna ca fiecare client are sanse mai mari sa cumpere, deci n estimat scade.
- Pentru Y = 0, posteriorul pentru n este mult mai mic cand theta este mare, deoarece este putin probabil sa avem multi clienti si totusi 0 cumparatori daca probabilitatea de cumparare este ridicata.

## c) Posterior predictiv pentru un numar viitor de cumparatori Y*
Am definit variabila y_new ~ Binomial(n, theta) si am folosit pm.sample_posterior_predictive pentru a genera mostre predictive pentru Y*. Rezultatele sunt plotate cu az.plot_dist (tot in format 3×2 subplot-uri). Aceste distributii arata cati cumparatori sunt de asteptat intr-o zi viitoare pe baza datelor observate.

## d) Diferenta dintre posterior si posteriorul predictiv
- Posteriorul estimeaza n (numarul real de clienti in ziua observata).
- Posteriorul predictiv estimeaza Y* (cati cumparatori vor fi intr-o zi viitoare).
- Posteriorul predictiv este mai lat deoarece combina incertitudinea asupra lui n cu variabilitatea binomiala a lui Y*.

In concluzie:
- Posteriorul raspunde la intrebarea: "Cat de probabil este fiecare n?"
- Posteriorul predictiv raspunde la intrebarea: "Cati cumparatori ne asteptam sa apara data viitoare?"

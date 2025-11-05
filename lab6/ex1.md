# Ex.1

P(B) = 0.01
P(T+ | B) = 0.95
P(T- | ¬B) = 0.90
P(T+ | ¬B) = 0.10

# a)

P(T+) = P(T+ | B) × P(B) + P(T+ | ¬B) × P(¬B) #formula prob totale
      = 0.95 × 0.01 + 0.10 × 0.99
      = 0.0095 + 0.099
      = 0.1085

P(B | T+) = P(T+ | B) × P(B) / P(T+) #Bayes
          = 0.0095 / 0.1085
          = 0.0876

Doar 8.76% probabilitate ca persoana sa aiba boala daca testul e pozitiv.

# b)

P(B | T+) = 0.5

s = specificitate
s = P(T- | ¬B), deci P(T+ | ¬B) = 1 - s

P(T+) = 0.95 × 0.01 + (1 - s) × 0.99
      = 0.0095 + 0.99 - 0.99s
      = 0.9995 - 0.99s

P(B | T+) = 0.0095 / (0.9995 - 0.99s) = 0.5

0.0095 = 0.5 × (0.9995 - 0.99s)
0.0095 = 0.49975 - 0.495s
0.495s = 0.49025
s = 0.9904

Specificitate minima: 99.04%

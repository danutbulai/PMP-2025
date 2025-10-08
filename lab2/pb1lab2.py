import random

# a)
def un_exp():
    u = ['R']*3 + ['B']*4 + ['K']*2
    d = random.randint(1,6)
    if d in [2,3,5]:
        u.append('K')
    elif d == 6:
        u.append('R')
    else:
        u.append('B')
    return random.choice(u)

# b)
def sim(n=100000):
    r = 0
    for i in range(n):
        if un_exp() == 'R':
            r += 1
    return r/n

p1 = sim(100000)
print("prob estimata:", round(p1,4))

# c)
pp = 3/6
p6 = 1/6
po = 2/6
r1 = 3/10
r2 = 4/10
r3 = 3/10

p2 = pp*r1 + p6*r2 + po*r3
print("prob teoretica:", round(p2,4))
print("diferenta:", round(abs(p1-p2),4))

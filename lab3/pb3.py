import numpy as np
from math import comb
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination

def sim(nr=10000):
    w0=w1=0
    for _ in range(nr):
        s=np.random.randint(0,2)
        n=np.random.randint(1,7)
        p=4/7 if s==0 else 1/2
        m=np.random.binomial(2*n,p)
        if n>=m:
            w0+=1 if s==0 else 0
            w1+=1 if s==1 else 0
        else:
            w0+=1 if s==1 else 0
            w1+=1 if s==0 else 0
    return w0/nr,w1/nr

g=DiscreteBayesianNetwork([('S','C'),('N','M'),('C','M')])

cpd_S=TabularCPD('S',2,[[0.5],[0.5]])
cpd_N=TabularCPD('N',6,[[1/6]]*6)
cpd_C=TabularCPD('C',2,[[0,1],[1,0]],evidence=['S'],evidence_card=[2])

vals=[]
for n in range(1,7):
    for c in [0,1]:
        p=0.5 if c==0 else 4/7
        col=[comb(2*n,k)*(p**k)*((1-p)**(2*n-k)) if k<=2*n else 0.0 for k in range(13)]
        s=sum(col)
        col=[x/s for x in col]
        vals.append(col)
vals=np.array(vals).T

cpd_M=TabularCPD('M',13,vals,evidence=['N','C'],evidence_card=[6,2])

g.add_cpds(cpd_S,cpd_N,cpd_C,cpd_M)
inf=VariableElimination(g)

if __name__=="__main__":
    p0,p1=sim(10000)
    print(f"sim_P0≈{p0:.4f}, sim_P1≈{p1:.4f}")
    ps=inf.query(['S'],evidence={'M':1}).values
    print(f"P(S=0|M=1)≈{ps[0]:.4f}, P(S=1|M=1)≈{ps[1]:.4f}")

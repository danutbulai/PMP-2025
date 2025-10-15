from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination

g = DiscreteBayesianNetwork([('S','O'),('S','L'),('S','M'),('L','M')])

cpd_S = TabularCPD('S',2,[[0.6],[0.4]])
cpd_O = TabularCPD('O',2,[[0.9,0.3],[0.1,0.7]],evidence=['S'],evidence_card=[2])
cpd_L = TabularCPD('L',2,[[0.7,0.2],[0.3,0.8]],evidence=['S'],evidence_card=[2])
cpd_M = TabularCPD('M',2,
    [[0.8,0.4,0.1,0.5],
     [0.2,0.6,0.9,0.5]],
    evidence=['S','L'],evidence_card=[2,2])

g.add_cpds(cpd_S,cpd_O,cpd_L,cpd_M)
inf = VariableElimination(g)

def clf(o,l,m):
    p = inf.query(['S'],evidence={'O':o,'L':l,'M':m}).values[1]
    y = int(p>=0.5)
    return p,y

if __name__ == "__main__":
    for o in [0,1]:
        for l in [0,1]:
            for m in [0,1]:
                p,y = clf(o,l,m)
                print(f"O={o}, L={l}, M={m} -> P(S=1|O,L,M)={p:.4f}, y={y}")

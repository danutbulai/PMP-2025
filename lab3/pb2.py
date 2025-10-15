from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination

g = DiscreteBayesianNetwork([('D','U'),('U','C')])

cpd_D = TabularCPD('D',6,[[1/6]]*6)
cpd_U = TabularCPD('U',3,
    [[0,0,0,0,0,1],
     [1,0,0,1,0,0],
     [0,1,1,0,1,0]],
    evidence=['D'],evidence_card=[6])

cpd_C = TabularCPD('C',3,
    [[0.4,0.3,0.3],
     [0.4,0.5,0.4],
     [0.2,0.2,0.3]],
    evidence=['U'],evidence_card=[3])

g.add_cpds(cpd_D,cpd_U,cpd_C)
inf = VariableElimination(g)
p = inf.query(['C']).values[0]
print(f"P(red)≈{p:.4f}")

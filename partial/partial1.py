import numpy as np
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination

g = DiscreteBayesianNetwork([('O','H'),('O','W'),('H','R'),('W','R'),('H','E'),('R','C')])
cpd_O = TabularCPD('O',2,[[0.3],[0.7]])
cpd_H = TabularCPD('H',2,[[0.9,0.2],[0.1,0.8]],evidence=['O'],evidence_card=[2])
cpd_W = TabularCPD('W',2,[[0.1,0.6],[0.9,0.4]],evidence=['O'],evidence_card=[2])
cpd_R = TabularCPD('R',2,[[0.6,0.9,0.3,0.5],[0.4,0.1,0.7,0.5]],evidence=['H','W'],evidence_card=[2,2])
cpd_E = TabularCPD('E',2,[[0.8,0.2],[0.2,0.8]],evidence=['H'],evidence_card=[2])
cpd_C = TabularCPD('C',2,[[0.85,0.4],[0.15,0.6]],evidence=['R'],evidence_card=[2])

g.add_cpds(cpd_O,cpd_H,cpd_W,cpd_R,cpd_E,cpd_C)
inf = VariableElimination(g)

p_H_given_C = inf.query(['H'],evidence={'C':0}).values
p_E_given_C = inf.query(['E'],evidence={'C':0}).values
print("b1)", round(p_H_given_C[0],4))
print("b2)", round(p_E_given_C[0],4))

q_hw = inf.query(['H','W'],evidence={'C':0}).values
idx = np.unravel_index(np.argmax(q_hw), q_hw.shape)
h_map = 'yes' if idx[0]==0 else 'no'
w_map = 'yes' if idx[1]==0 else 'no'
print("b3)", h_map, w_map, "prob", round(q_hw[idx],4))



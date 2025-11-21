import numpy as np
from hmmlearn.hmm import CategoricalHMM

pi = np.array([0.4, 0.3, 0.3])
A = np.array([[0.6, 0.3, 0.1],[0.2, 0.7, 0.1],[0.3, 0.2, 0.5],])
B = np.array([[0.1, 0.7, 0.2],[0.05, 0.25, 0.7],[0.8, 0.15, 0.05]])

m = CategoricalHMM(n_components=3, init_params="")
m.startprob_ = pi
m.transmat_ = A
m.emissionprob_ = B

obs_map = {'L':0,'M':1,'H':2}
states_map = ['W','R','S']
seq = ['M','H','L']
X = np.array([obs_map[o] for o in seq]).reshape(-1,1)

logp = m.score(X)
print("b)", np.exp(logp))

logp_v, z = m.decode(X, algorithm="viterbi")
path = [states_map[s] for s in z]
print("c) path:", path, "P(path,O)=", np.exp(logp_v))


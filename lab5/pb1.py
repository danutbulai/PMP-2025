import numpy as np
from hmmlearn.hmm import CategoricalHMM
import networkx as nx
import matplotlib.pyplot as plt

pi=np.array([1/3,1/3,1/3]) #vector de start
A=np.array([[0.0,0.5,0.5], #matricea de tranzitie
            [0.5,0.25,0.25],
            [0.5,0.25,0.25]])
B=np.array([[0.10,0.20,0.40,0.30], #matrice de emisii
            [0.15,0.25,0.50,0.10],
            [0.20,0.30,0.40,0.10]])

m=CategoricalHMM(n_components=3,init_params="") #hmm cu 3 parametri
m.startprob_=pi
m.transmat_=A
m.emissionprob_=B

lab={'FB':0,'B':1,'S':2,'NS':3}
seq=['FB','FB','S','B','B','S','B','B','NS','B','B']
X=np.array([lab[x] for x in seq]).reshape(-1,1) #vector de indici

lp=m.score(X) #calculeaza log-prob.
print("b) P(O)=",np.exp(lp)) #conversie

lp2,z=m.decode(X,algorithm="viterbi")
S=['D','M','E'] #starile 0,1,2
path=[S[i] for i in z]
print("c) path=",path)
print("c) P(path,O)=",np.exp(lp2))

G=nx.DiGraph()
for s in S: G.add_node(s)
for i in range(3):
    for j in range(3):
        if A[i,j]>0: G.add_edge(S[i],S[j],w=A[i,j])
p=nx.spring_layout(G,seed=0)
nx.draw(G,p,with_labels=True,node_size=1600,font_size=12,arrows=True)
el={(u,v):f"{d['w']:.2f}" for u,v,d in G.edges(data=True)}
nx.draw_networkx_edge_labels(G,p,edge_labels=el,font_size=10)
plt.title("a) HMM state diagram")
plt.tight_layout()
plt.savefig("hmm_states.png")
print("a) saved hmm_states.png")

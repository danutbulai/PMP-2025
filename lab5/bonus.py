import numpy as np

pi=np.array([1/3,1/3,1/3])
A=np.array([[0.0,0.5,0.5],
            [0.5,0.25,0.25],
            [0.5,0.25,0.25]])
B=np.array([[0.10,0.20,0.40,0.30],
            [0.15,0.25,0.50,0.10],
            [0.20,0.30,0.40,0.10]])

lab={'FB':0,'B':1,'S':2,'NS':3}
seq=['FB','FB','S','B','B','S','B','B','NS','B','B']
O=np.array([lab[x] for x in seq]) #vector de indici

T=len(O); N=3 #pasi si stari
d=np.zeros((T,N))
p=np.zeros((T,N),dtype=int)-1

d[0]=pi*B[:,O[0]] #initializare viterbi
for t in range(1,T):
    for j in range(N):
        v=d[t-1]*A[:,j]
        i=np.argmax(v)
        d[t,j]=v[i]*B[j,O[t]]
        p[t,j]=i

q=np.zeros(T,dtype=int) #vector cale optima
q[-1]=np.argmax(d[-1])
for t in range(T-2,-1,-1): #bkt
    q[t]=p[t+1,q[t+1]]

S=['D','M','E']
path=[S[i] for i in q]
Pstar=d[-1,q[-1]] #prob

print("path=",path)
print("P*(path,O)=",Pstar)

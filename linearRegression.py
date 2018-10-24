import numpy as np
import matplotlib.pyplot as plt

mul = np.matmul
inv = np.linalg.inv
rand = np.random.uniform

#array random di 20 elementi (training set)
X = rand(0,1, size = 20)

#array di 20 valori random tra -1 e 1
randArr = rand(-0.5, 0.5, size=20) 

#Y è un relazione lineare con X più un poò di rumore
Y = (rand(-3, 3)*X + rand(-3,3))
Y = Y + randArr
Y = Y.T

#metodo brutto per aggiungere gli 1 prima di ogni x in X
#non so come fare altrimenti
ones = np.ones((20,1))
X.shape = (20,1)
_X = []
for i, o in enumerate(ones):
    _X.append([])
    _X[i].extend(o)
    _X[i].extend(X[i])

X = np.array(_X)

#plotta i punti
for i, x in enumerate(X):
    plt.scatter(x[1], Y[i], marker=".", c="black")

#trasposta
X_t = X.T

#pseudoinversa
pseudoInv = mul( inv(mul(X_t, X)), X_t)

w = mul(pseudoInv, Y)

#w = [q, m] -> y = mx + q

print(w)

#                        [m*0 + q, m*1 + q]
line, = plt.plot([0, 1], [w[0], w[1] + w[0]])
plt.show()
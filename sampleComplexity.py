import sys
import math

script, Dvc, ep, delta = sys.argv

Dvc = int(Dvc)
ep = float(ep)
delta = float(delta)

print("VC Dimension : {}".format(Dvc))
print("Epsilon : {}".format(ep))
print("Delta : {}".format(delta))

def testN(N):
    poly = (2*N)**Dvc + 1
    internalLog = 4 * poly / delta
    ln = math.log(internalLog)
    rightSide = (8 / ep**2) * ln

    return N >= rightSide

#fare dicotomica
N = 2
base = N
top = N * 2
lastN = -1

i = 1

while lastN != N:
    i += 1
    lastN = N
    if not testN(N):
        base = N
        top = N * 2
    else:
        top = N

    N = math.ceil((base + top) / 2)

print("Minimal number of point is {}. Found after {} steps.".format(N, i))
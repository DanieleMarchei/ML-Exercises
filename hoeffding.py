import random
import sys
import math

if (len(sys.argv) == 4):
    script, N, ep, M = sys.argv
else:
    print("sample size, epsilon, M")
    sys.exit(0)

N = int(N)  #sample size
ep = float(ep)  #epsilon
M = int(M)  #numero di esperimenti (ipotesi)
mu = random.random()    #incognita

hoeff = float("{0:.5f}".format(2 * (math.e**(-2 * ep**2 * N) ) ))   #parte destra dell' ineguaglianza di hoeffding

print("N : ", N)
print("ep : ", ep)

print("Hoeffding : 2 * e^(-2 * {} * {}) = {}\n".format("{0:.5f}".format(ep**2), N, hoeff))

for i in range(M):
    nred = 0
    
    #prendo N palline dal bin con probabilità mu
    for p in range(N):
        if random.random() < mu:
            nred += 1
    
    #percentuale di palline rosse nel sample -> nred / N
    nu = nred / N

    print("{} -> nu = {}".format(i+1, nu))

    #i + 1 è il numero di esperimenti fatti
    print("P( | {} - mu | > {} ) < {}\n".format(nu, ep, (i+1) * hoeff))

import random
import sys
import math

if (len(sys.argv) > 2):
    script, N, e, M = sys.argv
else:
    print("sample size, epsilon, M")
    sys.exit(0)

N = int(N)
ep = float(e)
M = int(M)
mu = random.random()
hoeff = float("{0:.5f}".format(2 * 1 / (math.e**(2 * ep**2 * N) ) ))

print("N : ", N)
print("ep : ", ep)

print("Hoeffding : 2 * e^(-2 * {} * {}) = {}\n".format("{0:.5f}".format(ep**2), N, hoeff))

for i in range(M):
    palline = []
    ngreen = 0
    nred = 0
    sample = palline[0:N]
    for p in range(N):
        if random.random() < mu:
            sample.append("red")
            nred += 1
        else:
            sample.append("green")
            ngreen += 1


    selected = min(len(sample), 10)
    dots = ""
    if(len(sample) > 10):
        dots = "..."
    #print("samples = ",sample[0: selected], dots)

    new = nred / N
    print("{} -> nu = {}".format(i+1, new))

    print("P( | {} - mu | > {} ) < {}\n".format(new, ep, (i+1)*hoeff))

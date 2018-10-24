import numpy as np
import time
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import sys

fig, ax = plt.subplots()
xs = np.array([0,1])
X = np.array([[1, 0.1, 0.2], [1, 0.2, 0.1], [1, 0.15, 0.15], [1, 0.15, 0.2], # bottom-left corner
              [1, 0.3, 0.8], [1, 0.4, 1], [1, 0.35, 0.86], [1, 0.1, 0.9], # top-left corner
              [1, 0.45, 0.5], [1, 0.5, 0.6], [1, 0.4, 0.4], [1, 0.5, 0.4], # center
              [1, 0.6, 0.45], [1, 0.42, 0.46], [1, 0.6, 0.66], [1, 0.5, 0.3], # center
              [1, 0.8, 0.8], [1, 0.7, 0.6], [1, 0.9, 1], [1, 0.8, 0.9], # top-right corner
              [1, 0.7, 0.1], [1, 0.75, 0.3], [1, 0.9, 0.15], [1, 0.8, 0.35]]) # bottom-left corner

timeSleep = 0.05
if len(sys.argv) == 2 and sys.argv[1] == "-s":
    timeSleep = 0.5
    X = (X - 0.45) ** 2
    X = X * 4
    for i,x in enumerate(X):
        X[i][0] = 1
Y = np.array([-1, -1, -1, -1, -1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1, -1, -1])
w = np.array([-0.5, 0.5, 0.5])  #treshold, w1, w2

printedMessage = False

for i, x in enumerate(X):
    if Y[i] == -1:
        plt.scatter(x[1], x[2], marker="_", c="blue")
    else:
        plt.scatter(x[1], x[2], marker="^", c = "red")

def getLine():
    slope = -(w[0] / w[2]) / (w[0] / w[1])
    inter = -w[0] / w[2]
    return np.array([slope*0 + inter, slope*1 + inter])

line, = ax.plot([0,1], [getLine()[0], getLine()[1]])

def init():
    line.set_ydata(getLine())
    return line,

def getPoint(x):
    return w[0] * x[0] + w[1] * x[1] + w[2] * x[2]

def sign(x):
    if x < 0:
        return -1
    return 1

def learn():
    global w
    misclassified = None
    for i, x in enumerate(X):
        if sign(getPoint(x)) != Y[i]:
            misclassified = x, Y[i]
            break

    if misclassified != None:
        w = w + misclassified[1] * misclassified[0]
        w[0] = w[0] * misclassified[1]

    return misclassified == None

def animate(i):
    global printedMessage
    learned = learn()
    if not learned:
        line.set_ydata(getLine())
        print(w)
        time.sleep(timeSleep)
    elif not printedMessage:
        print("Best weights founded! -> {}  - after {} iterations".format(w, i))
        printedMessage = True
    return line,

ani = animation.FuncAnimation(
    fig, animate, init_func=init, interval=2, blit=True, save_count=50)

plt.show()

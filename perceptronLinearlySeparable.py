import numpy as np
import time
import matplotlib.pyplot as plt
import matplotlib.animation as animation

fig, ax = plt.subplots()
xs = np.array([0,1])
X = np.array([[1, 0.1, 0.2], [1, 0.1, 0.5], [1, 0.3, 0.4], [1, 0.4, 0.1],
              [1, 0.2, 0.3], [1, 0.32, 0.2], [1, 0, 0.4], [1, 0.5, 0.4],
              [1, 0.4, 0.7], [1, 0.7, 0.2], [1, 0.7, 0.4], [1, 0.8, 0.6],
              [1, 0.5, 0.8], [1, 0.75, 0.5], [1, 0.63, 0.6], [1, 0.55, 0.67]])
Y = np.array([-1, -1, -1, -1, -1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, 1])
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

    return misclassified == None

def animate(i):
    global printedMessage
    learned = learn()
    if not learned:
        line.set_ydata(getLine())
        print(w)
        time.sleep(0.2)
    elif not printedMessage:
        print("Best weights founded! -> {}".format(w))
        printedMessage = True
    return line,

ani = animation.FuncAnimation(
    fig, animate, init_func=init, interval=2, blit=True, save_count=50)

plt.show()

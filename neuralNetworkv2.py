import math
import random as rnd
from PIL import Image

w, h = 100, 100

dataset = {
    "input" : [[0,0], [0,1], [1,0], [1,1]],
    "output": [0, 1, 1, 0]
}

learningRate = 1

inputs = 2
hiddenNodes = 10
outputs = 1

#[[w11, w12, w13] , [w21, w22, w23]]
inputWeights = [[0]*hiddenNodes]*inputs
hiddenBias = [0]*hiddenNodes

#[[w14, w24, w34]]
outputWeights = [[0]*hiddenNodes]*outputs
outputBias = [0]*outputs

z = [0] * (hiddenNodes + outputs)

def initilize():
    for i in range(len(inputWeights)):
        for j in range(len(inputWeights[i])):
            inputWeights[i][j] = rnd.random()

    for i in range(len(hiddenBias)):
        hiddenBias[i] = rnd.random()

    for i in range(len(outputWeights)):
        outputWeights[0][i] = rnd.random()
    
    outputBias[0] = rnd.random()

def sigmoid(x, deriv = None):
    if deriv == None:
        return 1/(1 + math.e**(-x))
    else:
        sig_x = sigmoid(x)
        return sig_x * (1 - sig_x)

def tanh(x, deriv = None):
    if deriv == None:
        return (math.e ** x - math.e ** (-x))/(math.e ** x + math.e ** (-x))
    else:
        return 1 - tanh(x)**2

def relu(x, deriv = None):
    if deriv == None:
        return max(0, x)
    else:
        return 1 if x > 0 else 0

def feedForward(x):
    global z
    for n in range(hiddenNodes):
        z[n] = hiddenBias[n]
        for i in range(inputs):
            z[n] += x[i] * inputWeights[i][n]

    z[-1] = outputBias[0]
    for n in range(hiddenNodes):
        z[-1] += activation(z[n]) * outputWeights[0][n]

    return sigmoid(z[-1])

def backProp(predicted, target, x):
    delta = (predicted - target) * activation(z[-1], True)

    for n in range(hiddenNodes):
        err = delta * sigmoid(z[n])
        outputWeights[0][n] -= learningRate * err

    for o in range(outputs):
        outputBias[o] -= learningRate * delta

    for n in range(hiddenNodes):
        for i in range(inputs):
            err = x[i] * activation(z[n], True) * delta * outputWeights[0][n]
            inputWeights[i][n] -= learningRate * err 

    for n in range(hiddenNodes):
        hiddenBias[n] -= activation(z[n], True) * delta * outputWeights[0][n]

initilize()

activation = tanh

iterations = 100000
for i in range(iterations):
    index = rnd.randint(0, 3)
    x = dataset["input"][index]
    target = dataset["output"][index]
    predict = feedForward(x)
    backProp(predict, target, x)

print "--------------- TRAINED"
for k,v in enumerate(dataset["input"]):
    print v , "->\t" , feedForward(v)

img = Image.new('RGB', (w, h))
pixels = img.getpixel
for i in range(w):
    for j in range(h):
        _i = i / float(w)
        _j = j / float(h)
        p = feedForward([_i, _j])
        _p = int(p * 255)

        img.putpixel((i, j) , (_p, _p, _p))



img.save("img.bmp")
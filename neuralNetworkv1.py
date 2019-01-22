import math
import random as rnd
from PIL import Image

# 2 inputs, 1 hidden layer with 4 neurons, 1 output

dataset = {
    "input" : [[0,0], [0,1], [1,0], [1,1]],
    "output": [0, 1, 1, 0]
}

learningRate = 0.5

#[[w11, w12, w13] , [w21, w22, w23]]
inputWeights = [[0,0,0] , [0,0,0]]
hiddenBias = [0,0,0]

#[[w14, w24, w34]]
outputWeights = [[0,0,0]]
outputBias = [0]

z1 = 0
z2 = 0
z3 = 0
z4 = 0

def initilize():
    for i in range(2):
        for j in range(3):
            inputWeights[i][j] = rnd.random()

    for i in range(3):
        hiddenBias[i] = rnd.random()

    for i in range(3):
        outputWeights[0][i] = rnd.random()
    
    outputBias[0] = rnd.random()

def sigmoid(x):
    return 1/(1 + math.e**(-x))

def sigmoidDeriv(x):
    sig_x = sigmoid(x)
    return sig_x * (1 - sig_x)

def feedForward(x):
    global z1, z2, z3, z4
    z1 = hiddenBias[0] + x[0]*inputWeights[0][0] + x[1]*inputWeights[1][0]
    z2 = hiddenBias[1] + x[0]*inputWeights[0][1] + x[1]*inputWeights[1][1]
    z3 = hiddenBias[2] + x[0]*inputWeights[0][2] + x[1]*inputWeights[1][2]

    sig_z1 = sigmoid(z1)
    sig_z2 = sigmoid(z2)
    sig_z3 = sigmoid(z3)

    z4 = outputBias[0] + sig_z1*outputWeights[0][0] + sig_z2*outputWeights[0][1] + sig_z3*outputWeights[0][2]

    return sigmoid(z4)

def backProp(predicted, target, x):
    delta = (predicted - target) * sigmoidDeriv(z4)

    err_w14 = delta * sigmoid(z1)
    outputWeights[0][0] -= learningRate * err_w14

    err_w24 = delta * sigmoid(z2)
    outputWeights[0][1] -= learningRate * err_w24

    err_w34 = delta * sigmoid(z3)
    outputWeights[0][2] -= learningRate * err_w34

    outputBias[0] -= learningRate * delta

    err_w11 = x[0] * sigmoidDeriv(z1) * delta * outputWeights[0][0]
    err_w21 = x[1] * sigmoidDeriv(z1) * delta * outputWeights[0][0]
    inputWeights[0][0] -= learningRate * err_w11
    inputWeights[1][0] -= learningRate * err_w21

    err_w12 = x[0] * sigmoidDeriv(z2) * delta * outputWeights[0][1]
    err_w22 = x[1] * sigmoidDeriv(z2) * delta * outputWeights[0][1]
    inputWeights[0][1] -= learningRate * err_w12
    inputWeights[1][1] -= learningRate * err_w22

    err_w13 = x[0] * sigmoidDeriv(z3) * delta * outputWeights[0][2]
    err_w23 = x[1] * sigmoidDeriv(z3) * delta * outputWeights[0][2]
    inputWeights[0][2] -= learningRate * err_w13
    inputWeights[1][2] -= learningRate * err_w23

    hiddenBias[0] -= sigmoidDeriv(z1) * delta * outputWeights[0][0]
    hiddenBias[1] -= sigmoidDeriv(z2) * delta * outputWeights[0][1]
    hiddenBias[2] -= sigmoidDeriv(z3) * delta * outputWeights[0][2]

initilize()

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

w, h = 600, 600

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
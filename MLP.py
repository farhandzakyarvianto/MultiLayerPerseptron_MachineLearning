import math
import random
import numpy as np

#Data
wh=np.random.uniform(-1,1,size=(2,3))
vh=np.random.uniform(-1,1,size=(1,3))
lr = 0.03

print("Awal")
print('W : ' + str(wh))
print('V : ' + str(vh))
print('--------------------------------------------------------------')
print('--------------------------------------------------------------')

datasets = [[4, 0, 1], [0, 4, 1], [4, 8, 1], [8, 4, 1],
           [4, 2, 0], [2, 4, 0], [6, 4, 0], [4, 6, 0]]


epoch = 5000
E = []
step = 0

def sigmoid(x):
    return ( 1 / (1 + math.exp(-x)))

# mencari y
y=[]

for i in range(len(datasets)):

    zh1 = sigmoid(wh[0][0] + wh[0][1] * datasets[i][0] + wh[0][2] * datasets[i][1])
    zh2 = sigmoid(wh[1][0] + wh[1][1] * datasets[i][0] + wh[1][2] * datasets[i][1])

    y.append(vh[0][0] + (vh[0][1] * zh1) + (vh[0][2] * zh2))

    for j in range(epoch):
        y[i] = vh[0][0] + (vh[0][1] * zh1) + (vh[0][2] * zh2)

        vh[0][0] = vh[0][0] + lr * (datasets[i][2] - sigmoid(y[i])) * sigmoid(y[i]) * (1 - sigmoid(y[i]))
        vh[0][1] = vh[0][1] + lr * (datasets[i][2] - sigmoid(y[i])) * sigmoid(y[i]) * (1 - sigmoid(y[i])) * zh1
        vh[0][2] = vh[0][2] + lr * (datasets[i][2] - sigmoid(y[i])) * sigmoid(y[i]) * (1 - sigmoid(y[i])) * zh2

        wh[0][0] = wh[0][0] + lr * (datasets[i][2] - sigmoid(y[i])) * sigmoid(y[i]) * (1 - sigmoid(y[i])) * vh[0][
            1] * sigmoid(zh1) * (1 - sigmoid(zh1))
        wh[0][1] = wh[0][1] + lr * (datasets[i][2] - sigmoid(y[i])) * sigmoid(y[i]) * (1 - sigmoid(y[i])) * vh[0][
            1] * sigmoid(zh1) * (1 - sigmoid(zh1)) * datasets[i][0]
        wh[0][2] = wh[0][2] + lr * (datasets[i][2] - sigmoid(y[i])) * sigmoid(y[i]) * (1 - sigmoid(y[i])) * vh[0][
            1] * sigmoid(zh1) * (1 - sigmoid(zh1)) * datasets[i][1]

        wh[1][0] = wh[1][0] + lr * (datasets[i][2] - sigmoid(y[i])) * sigmoid(y[i]) * (1 - sigmoid(y[i])) * vh[0][
            2] * sigmoid(zh2) * (1 - sigmoid(zh2))
        wh[1][1] = wh[1][1] + lr * (datasets[i][2] - sigmoid(y[i])) * sigmoid(y[i]) * (1 - sigmoid(y[i])) * vh[0][
            2] * sigmoid(zh2) * (1 - sigmoid(zh2)) * datasets[i][0]
        wh[1][2] = wh[1][2] + lr * (datasets[i][2] - sigmoid(y[i])) * sigmoid(y[i]) * (1 - sigmoid(y[i])) * vh[0][
            2] * sigmoid(zh2) * (1 - sigmoid(zh2)) * datasets[i][1]

        error = datasets[i][2] - sigmoid(y[i])
        E.append(error)
    while step < epoch:
        print('epoch : ' + str(step) + ', error : ' + str(E[step]))
        step += 1


print('--------------------------------------------------------------')
print('--------------------------------------------------------------')

print("akhir")
print('W : ' + str(wh))
print('V : ' + str(vh))
print('--------------------------------------------------------------')
for k in range(len(datasets)):
    if sigmoid(y[k]) > 0.5:
        th = 1
    else:
        th = 0
    print('prediksi :' + str(datasets[k][2]) + ', result : ' + str(sigmoid(y[k])))


# import matplotlib
# import matplotlib.pyplot as plt
# plt.xlabel('epoch')
# plt.plot(E[:5000])
# plt.ylabel('error')
# plt.show()

print('--------------------------------------------------------------')

datatest = [[-8, 8, 0]]

g = []

for i in range(len(datatest)):
    zh1 = sigmoid(wh[0][0] + wh[0][1] * datatest[i][0] + wh[0][2] * datatest[i][1])
    zh2 = sigmoid(wh[1][0] + wh[1][1] * datatest[i][0] + wh[1][2] * datatest[i][1])

    g.append(sigmoid(vh[0][0] + (vh[0][1] * zh1) + (vh[0][2] * zh2)))

for k in range(len(datatest)):
    if g[k] > 0.5:
        th = 1
    else:
        th = 0
    print('prediksi :%d, result :%.3f' % (datatest[k][2], g[k]))
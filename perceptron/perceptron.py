import os
import perceptron
import numpy as np
import matplotlib.pyplot as plt

class Perceptron:
    def __init__(self):
        np.random.seed(100)
       # self.weight = [0.0,0.0]
        self.weight=np.random.rand(2)
        self.bias = np.random.rand(1)
 
def activatorRelu(val):
    if val>0:
        return 1
    else:
        return 0
def updateWeight(label,weight,bias,rate,output,input):
    delta = label - output
    w1 = delta*input[0]*rate+weight[0]
    w2 = delta*input[1]*rate+weight[1]
    bias = bias+rate*delta
    return delta,[w1,w2],bias
 
def calcOutput(x1,x2,w1,w2,b):
    return x1*w1+x2*w2+b
 
if __name__=="__main__":
    #初始化，赋初值
    p1 = perceptron.Perceptron()
    print(p1.weight)
    print(p1.bias)
    list_w0=[]
    list_w1=[]
    list_b=[]
    out=[]
    input = [[0,0],[0,1],[1,0],[1,1]]
    label = [0,0,0,1]
    learnRate = 0.01
    lun=0
    #训练迭代，更新权重、偏置
    while lun<=100:
        for x in input:
            print("lun:",lun,x)
            output = calcOutput(x[0],x[1],p1.weight[0],p1.weight[1],p1.bias)

            print("before activate output:",output)
            output = activatorRelu(output)
            
            print("after activate output:",output)
            # raise "\nend****"
            print("label:",label[input.index(x)],"index:",input.index(x))
            delt,p1w,p1b = updateWeight(label[input.index(x)],p1.weight,p1.bias,learnRate,output,x)
            p1.weight,p1.bias = p1w,p1b
            
            list_w0.append(p1.weight[0])
            list_w1.append(p1.weight[1])
            list_b.append(p1.bias)
            print(p1.weight,p1.bias)
            print("\n******************************************\n")
        lun = lun + 1

    # plt.plot(list_w0,color='r')
    # plt.plot(list_w1,color='g')
    # plt.plot(list_b,linestyle='--')
    # plt.title("w0:red     w1: greed     b:--",fontsize=10)
    plt.xlim([0,200])

    plt.show()
        #预测过程
    # preInput = [0,1]
    # output = calcOutput(preInput[0],preInput[1],p1.weight[0],p1.weight[1],p1.bias)
    # print("predict output:",output)
    # output = activatorRelu(output)
    # print("predict output:",output)
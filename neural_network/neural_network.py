import numpy as np
import pandas as  pd
import scipy.optimize as opt
import matplotlib.pyplot as plt
import scipy.io as sio
from sklearn.metrics import classification_report

def load_data(path, transpose=True):
    data = sio.loadmat(path)
    y = data.get('y')  # (5000,1)
    y = y.reshape(y.shape[0])  # make it back to column vector

    X = data.get('X')  # (5000,400)

    if transpose:
        # for this dataset, you need a transpose to get the orientation right
        X = np.array([im.reshape((20, 20)).T for im in X])

        # and I flat the image again to preserve the vector presentation
        X = np.array([im.reshape(400) for im in X])

    return X, y
X, y = load_data('.vscode\ex3data1.mat')
# print(data)
# print(data['X'].shape)
# print(data['y'].shape)
print(X.shape)
print(y.shape)


def plot_an_image(image):
#     """
#     image : (400,)
#     """
    fig, ax = plt.subplots(figsize=(1, 1))
    ax.matshow(image.reshape((20, 20)), cmap=plt.cm.binary)
    plt.xticks(np.array([]))  # 去掉刻度，因为matshow方法绘制矩阵图会自带刻度
    plt.yticks(np.array([]))
#绘图函数
# pick_one = np.random.randint(0, 5000)#返回[0，5000)中任意一个整数
# plot_an_image(X[pick_one, :])  #取X的一行  有400个特征
# plt.show()
# print('this should be {}'.format(y[pick_one]))

#绘制数据集中随机100个图片
def plot_100_image(X):

    size=int(np.sqrt(X.shape[1]))
    pick_100=np.random.choice(np.arange(X.shape[0]),100)
    pick_image=X[pick_100,:]
    fig,axs=plt.subplots(10,10,sharex=True,sharey=True,figsize=(8,8))
    for r in range(10):
        for c in range(10):
            axs[r,c].matshow(pick_image[r*10+c].reshape(size,size),cmap=plt.cm.binary)
            plt.xticks(np.array([]))
            plt.yticks(np.array([]))
# plot_100_image(X)
# plt.show()
data_X = np.insert(X, 0, values=np.ones(X.shape[0]), axis=1) 
#等价于
# x=pd.DataFrame(X)
# ones=pd.DataFrame(np.ones([X.shape[0],1]))
# data_X=pd.concat([ones,x],axis=1)
# print(data_X.iloc[:,:].values)

y_matrix=[]
for k in range(1,11):
    y_matrix.append((y==k).astype(int))#条件成立 赋值1，否则赋值0

y_matrix=[y_matrix[-1]]+y_matrix[:-1]

y_matrix=np.array(y_matrix)
print("label:",y_matrix.shape)
print(y_matrix)
#对于普通一维模型的训练 即参数为 1*n
def sigmoid(z):
    return 1/(1+np.exp(-z))
def cost(theta,X,y):
    return np.mean(-y*np.log(sigmoid(X@theta))-(1-y)*np.log(1-sigmoid(X@theta)))
def regularized_cost(theta,X,y,l):
    theta_j=theta[1:]
    regularize_theta=(1/2*len(X))*(np.power(theta_j,2)).sum()
    return cost(theta,X,y)+regularize_theta

def gradient(theta,X,y):
    return (1/len(X))*X.T@(sigmoid(X@theta)-y)
    
def regularized_gradient(theta, X, y, l=1):
    '''still, leave theta_0 alone'''
    theta_j1_to_n = theta[1:]
    regularized_theta = (l / len(X)) * theta_j1_to_n

    # by doing this, no offset is on theta_0
    regularized_term = np.concatenate([np.array([0]), regularized_theta])

    return gradient(theta, X, y) + regularized_term
def logistic_regression(X,y,l=1):
    """
    X为m*n+1的矩阵， x0=1  (m,n+1)
    y是一个列向量(m,)
    l 是正则化参数
    """
    theta=np.zeros(X.shape[1])
    res=opt.minimize(fun=regularized_cost,x0=theta,args=(X,y,l),jac=regularized_gradient,method='TNC')
    return res.x
def predict(X,theta):
    prob=sigmoid(X@theta)
    return (prob>=0.5).astype(int)
#预测其中一个的结果
t0=logistic_regression(data_X,y_matrix[0])
y_pred=predict(data_X,t0)
print("y_pred:",y_pred)
print('Accuracy={}'.format(np.mean(y_matrix[0] == y_pred)))

#本次是0-9  十分类问题，传统的一维模型将变成K维模型
#初始化
k_theta=np.array([logistic_regression(data_X,y_matrix[k]) for k in range(10)])
print(k_theta.shape)
print(k_theta)
#进行预测
#data_X：(5000,401), k_theta:(10,401),y_matrix:(10,5000)
#将做 data_X@ k_theta.T 的运算 
#prob_matrix=sigmoid(data_X@k_theta.T)
prob_matrix=(sigmoid(data_X@k_theta.T)>=0.5).astype(int)
#np.set_printoptions(suppress=True)
print(prob_matrix.shape)
print(prob_matrix)
print(classification_report(y_matrix.T, prob_matrix))

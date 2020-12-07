import pandas as pd 
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt 
plt.style.use('bmh')
from sklearn.metrics import classification_report  #得出准确率，召回率，F1率
import scipy.optimize as opt


#读取数据
data= pd.read_csv('.vscode\ex2data1.txt',names=['exam1','exam2','admitted'])
print(data.shape)
#print(data.head())
sns.set(context="notebook", style="darkgrid")

sns.relplot(x='exam1', y='exam2', hue="admitted",style="admitted" ,data=data)

#读取特征
# ones=pd.DataFrame(np.ones([10,1]),columns=['sds'])
# print(ones)
def get_X(data):
    ones=pd.DataFrame(np.ones([data.shape[0],1]),columns=['ones'])
    data=pd.concat([ones,data],axis=1)
    return data.iloc[:,:-1].values
def get_y(data):
    return np.array(data.iloc[:,-1])

#标准化，提高梯度下降的效率
def normalizer(data):
    return data.apply(lambda column: (column-column.mean())/column.std())

data_x=get_X(data)
data_y=get_y(data)
# print(data_x.shape)
# print(data_y.shape)
#weights=np.random.randn(3)
weights=np.zeros(data.shape[1])


def sigmoid(z):
    return 1/(1+np.exp(-z))

def cost_function(w, x, y,l=0):#注意参数的顺序，和minimize函数的参数顺序要一致
    theta_j1_to_n = w[1:]
    regularized_term = (l / (2 * len(x))) * np.power(theta_j1_to_n, 2).sum()
    return np.mean(-y * np.log(sigmoid(x @ w)) - (1 - y) * np.log(1 - sigmoid(x @ w)))+regularized_term
#print(cost_function(data_x,data_y,weights))
def gradient(w,x,y,l=0):
#     '''just 1 batch gradient'''
    theta_j1_to_n = w[1:]
    regularized_theta = (l / len(x)) * theta_j1_to_n   
    regularized_term = np.concatenate([np.array([0]), regularized_theta]) 
    return (1 / len(x)) * x.T @ (sigmoid(x @ w) - y)+ regularized_term
#print(gradient(data_x,data_y,weights))
ans=opt.minimize(fun=cost_function,x0=weights,args=(data_x,data_y),method='Newton-CG',jac=gradient)
print(ans)

#验证
def predict(x,w):
    ans=sigmoid(x@w)
    return (ans>=0.5).astype(int)
y_pred=predict(data_x,ans.x)
print(classification_report(data_y,y_pred))
x=np.arange(100,step=0.1)
y=-ans.x[0]/ans.x[2]-(ans.x[1]/ans.x[2])*x
plt.plot(x,y)
plt.xlim(30, 100)
plt.ylim(30, 100)
plt.show()
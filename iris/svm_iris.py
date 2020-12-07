import numpy as np 
import pandas as pd 
from sklearn import svm
from sklearn.svm import SVC
from sklearn import model_selection
import seaborn as sns
import matplotlib.pyplot as plt
plt.style.use('bmh')
#*************将字符串转为整型，便于数据加载***********************
# def iris_type(s):
#     it = {'Iris-setosa':0, 'Iris-versicolor':1, 'Iris-virginica':2}
#     return it[s]
data=pd.read_csv('.vscode\iris.data',names=['sepal_length','sepal_width','huaban_len','huaban_width','fenlei']
                #,converters={4:iris_type}
                )
print(data)
sns.set(context="notebook",style="darkgrid")
sns.relplot(x='sepal_length',y='sepal_width',hue="fenlei",style="fenlei",data=data)
#数据可视化
#plt.show()

def get_X(data):
    return np.array(data.iloc[:,0:2])
def get_y(data):
    return np.array(data.iloc[:,-1])
x=get_X(data)
y=get_y(data)
#print(x.shape)  
x_train,x_test,y_train,y_test=model_selection.train_test_split(x,y,random_state=1,test_size=0.3)

#模型搭建;

def classify():
    #clf = svm.SVC(C=0.8,kernel='rbf', gamma=50,decision_function_shape='ovr')
    clf = svm.SVC(C=0.5,                         #误差项惩罚系数,默认值是1
                  kernel='linear',               #线性核 kenrel="rbf":高斯核
                  decision_function_shape='ovr') #决策函数
    return clf

clf=classify()

#模型训练
def train(clf,x_train,y_train):
    clf.fit(x_train,                    #训练集特征
            y_train.ravel())            #目标值
train(clf,x_train,y_train)
clf_train=clf.predict(x_train)
clf_test=clf.predict(x_test)

print(clf.score(x_test,y_test))
a=clf.coef_
b=clf.intercept_
# print(a)
# print(b)
x1=np.arange(10,step=0.1)
#s三个二分类
# y1=((-a[0][0])*x1-b[0])/a[0][1]
# y2=((-a[1][0])*x1-b[1])/a[1][1]
# y3=((-a[2][0])*x1-b[2])/a[2][1]
# plt.plot(x1,y1)
# plt.plot(x1,y2)
# plt.plot(x1,y3)
# plt.xlim(4,7)
# plt.ylim(2,5)

#画决策边界
def plot_decision_boundary(pred_func):
 
    # 设定最大最小值，附加一点点边缘填充
    x_min, x_max = x[:, 0].min() - .5, x[:, 0].max() + .5
    y_min, y_max = x[:, 1].min() - .5, x[:, 1].max() + .5
    h = 0.01
 
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
 
    # 用预测函数预测一下
    Z = pred_func(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
 
    # 然后画出图
    plt.contourf(xx, yy, Z,cmap='GnBu')
    plt.scatter(x[:, 0], x[:, 1], c=y)

plot_decision_boundary(lambda x: clf.predict(x))
plt.show()
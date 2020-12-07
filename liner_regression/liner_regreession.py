import numpy as np
import pandas as pd 
import seaborn as sns
sns.set(context="notebook",style="whitegrid",palette="dark")
import matplotlib.pyplot as plt 
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

#pandas 读取数据
df=pd.read_csv('.vscode\ex1data1.txt',names=['population','profit'])
#print(df.head())
#sns.lmplot(x='population',y='profit',data=df)

#print(df.describe())

#读取数据的特征
def get_X(df):
    #n行1列的全一矩阵，x0=1
    #ones=pd.DataFrame({'ones':np.ones(len(df))})
    #数据整个为{x0,x1,x2....xn}
   # data=pd.concat([ones,df],axis=1)
    #return data.iloc[:,:-1].values  #as_matrix()将DataFrame类型的数据转换为narray
    return np.array(df.iloc[:,:-1])

#读取标签y
def get_y(df):
    return np.array(df.iloc[:,-1])
#对数据进行归一化处理（x-均值）/标准差
def normalize(data):
    mean=np.mean(data)
    std=np.std(data)
    data=(data-mean)/std
    return data

def normalize_feature(df):
    return df.apply(lambda column:(column - column.mean()) / column.std())

#线性回归
def liner_RE(x_data,y_data,epoch):
    x=tf.placeholder(tf.float32,name='x',shape=x_data.shape)
    y=tf.placeholder(tf.float32,name='y',shape=y_data.shape)
    m=x_data.shape[0]
    n=x_data.shape[1]
    w=tf.Variable(tf.zeros([n,1]),name='w')
    b = tf.Variable(0.0,name='b')
    y_pred=tf.matmul(x,w)+b
    loss=tf.reduce_mean(tf.square(y_pred-y,name='loss'))
    optimizer=tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(loss)
    init_op=tf.global_variables_initializer()
    total=[]
    #训练过程：
    with tf.Session() as sess:
        sess.run(init_op)
        writer=tf.summary.FileWriter('graphs',sess.graph)
        for i in range(epoch):
            _,l=sess.run([optimizer,loss],feed_dict={x:x_data,y:y_data})
            total.append(l)
            print('Epoch{0}:loss{1}'.format(i,l))
            writer.close()
            w_value,bias=sess.run([w,b])
        
        
        plt.scatter(df.population, df.profit, label="Training data")
        plt.plot(df.population, df.population*w_value[0][0] + bias, label="Prediction")
        print(w_value[0][0],bias)
        #print(w_value)
        #plt.plot(total)
        #plt.show()
        return w_value


normalize_feature(df)
x_data=get_X(df)
y_data=get_y(df)
print(x_data.shape[1])
alpha=0.01
epoch=50
liner_RE(x_data,y_data,epoch)

plt.show()


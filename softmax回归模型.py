import pandas as pd
import numpy as np

def one_hot(label,n,k):
    '''形成one hot矩阵'''

    one_hot=np.zeros([k,n])
    one_hot[label,np.arange(n)]=1
    return one_hot

def Normalize(data):
    '''归一化'''

    #m=np.mean(data)
    #mx=np.max(data)
    #mn=np.min(data)
    #return (data-m)/(mx-mn)
    return data/255

def softmax(data):
    '''softmax层输出'''

    m=np.sum(data,axis=0)
    return data/m

def labelfun(data):
    '''得到标签'''

    for u in range(10):
        if data[u]==np.max(data):
            s=u
    return s

def gradient_descent(n,alpha,lamda,arr1,arr2,label):
    '''梯度下降'''

    s=0
    for i in range(int(n)):
        arr=np.dot(arr1,arr2.T)
        for l in range(20000):#减去最大值防止e^溢出
            arr[:,l]=arr[:,l]-np.max(arr[:,l])
        h_theta=np.exp(arr)
        for i in range(0,20000):#得到对应标签的概率
            h_theta[:,i]=softmax(h_theta[:,i])
        h_theta=one_hot(label,20000,10)-h_theta#得到1{yi=j}-h_theta[i,k]的对应矩阵
        grad=-1/20000*np.dot(h_theta,arr2)+lamda*arr1#偏导数
        arr1=arr1-alpha*grad
        s+=1
        print('gradient',grad)
        print(s)#迭代次数
    return arr1

def forecast(arr1,arr2):
    '''预测结果'''
    for i in range(1,785):#测试集归一化
        if (np.array(arr2[:,i])==0).all()!=True:
            arr2[:,i]=Normalize(arr2[:,i])
    arr=np.dot(arr1,arr2.T)
    for l in range(10000):#减去最大值防止溢出
        arr[:,l]=arr[:,l]-np.max(arr[:,l],axis=0)
    h_theta=np.exp(arr)
    for i in range(0,10000):#得到对应标签的概率
        h_theta[:,i]=softmax(h_theta[:,i])
    labelout=np.zeros(10000)
    for i in range(9999):
        labelout[i]=labelfun(h_theta[:,i+1])
    labelout[9999]=labelfun(h_theta[:,0])
    return labelout



#计算模型
train_filename='C:\\Users\\86138\\Desktop\\分类\\train.csv'
train_df=pd.read_csv(train_filename)
train_df['id']=1#第一列全部替换为1
theta=np.random.random((10,785))#模型参数
theta_filename='C:\\Users\\86138\\Desktop\\theta.csv'
#theta=np.loadtxt(theta_filename)
label_df=train_df['label']
label=label_df.values#转化为数组
del train_df['label']
train=(train_df.values).astype(float)
for i in range(1,785):#归一化
    if (np.array(train[:,i])==0).all()!=True:#不计算全为0的列
        train[:,i]=Normalize(train[:,i])
print('输入迭代次数，学习率以及正则化系数')
a,b,c=map(float,input().split())
theta=gradient_descent(a,b,c,theta,train,label)
np.savetxt('C:\\Users\\86138\\Desktop\\theta.csv',theta,fmt='%f')#保存参数



#预测结果
test_filename='C:\\Users\\86138\\Desktop\\分类\\test.csv'
test_df=pd.read_csv(test_filename)
test_df['id']=1#第一列全部替换为1
test=(test_df.values).astype(float)
out=forecast(theta,test)
id=np.arange(1,10001)
id[9999]=0
submission=pd.DataFrame({'id':id,'label':out})
submission.to_csv("C:\\Users\\86138\\Desktop\\submission.csv",index=0)

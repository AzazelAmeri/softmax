import pandas as pd
import numpy as np

#示性函数
def indic(a,b):
    if a==b:
        return 1
    else:
        return 0

#归一化
def Normalize(data):
    m=np.mean(data)
    mx=np.max(data)
    mn=np.min(data)
    return (data-m)/(mx-mn)

#输出概率
def softmax(data):
    m=np.sum(data,axis=0)
    return data/m

#得到标签
def labelfun(data):
    for u in range(10):
        if data[u]==np.max(data):
            s=u
    return s

####向量化
filename='C:\\Users\\86138\\Desktop\\分类\\train.csv'
train_df=pd.read_csv(filename)
train_df['id']=1#第一列全部替换为1
theta=np.random.random((10,785))#模型参数
#theta=np.loadtxt('C:\\Users\\86138\\Desktop\\theta.csv')#接着上次的结果继续算
label_df=train_df['label']
label=label_df.values#转化为数组
del train_df['label']
train=(train_df.values).astype(float)
for i in range(1,785):#归一化
    if (np.array(train[:,i])==0).all()!=True:#不计算全为0的列
        train[:,i]=Normalize(train[:,i])

num=0
##批量梯度下降
for i in range(1000):
    arr=np.dot(theta,train.T)
    for l in range(20000):#减去最大值防止e^z溢出
        arr[:,l]=arr[:,l]-np.max(arr[:,l])
    h_theta=np.exp(arr)
    for i in range(0,20000):#得到对应标签的概率
        h_theta[:,i]=softmax(h_theta[:,i])
    for i in range(10):
        for k in range(20000):
            h_theta[i,k]=indic(label[k],i)-h_theta[i,k]#得到1{yi=j}-h_theta[i,k]的对应矩阵
    dir=-1/20000*np.dot(h_theta,train)+0.0001*theta#偏导数
    theta=theta-0.1*dir#theta-偏导数
    num+=1
    print('dir',dir)
    print(num)#迭代次数

###随机梯度下降
#for i in range(50000) :
#    k=np.random.randint(0,10000)
#    arr=np.dot(theta,train[[k],:].T)
#    arr=arr-np.max(arr)#theta x-xi最大值防止e^z溢出
#    h_theta=np.exp(arr)
#    h_theta=Normalizeout(h_theta)#得到对应标签的概率
#    matri=h_theta[:,:]
#    for i in range(10):
#        matri[i]=indic(label[k],i)-h_theta[i]#得到1{yi=j}-h_theta[i,k]的对应矩阵
#    dir=-np.dot(matri,train[[k],:])+0.0003*theta
#    theta=theta-0.3*dir#theta-偏导数
#    print('dir',dir)
#    #print('theta',theta)
#    #if((np.array(np.abs(dir))<0.01).all()and(np.array(np.abs(dir))>-0.01).all()):break

np.savetxt('C:\\Users\\86138\\Desktop\\theta.csv',theta,fmt='%f')#保存参数

###预测结果
theta2=np.loadtxt('C:\\Users\\86138\\Desktop\\theta.csv')
filename2='C:\\Users\\86138\\Desktop\\分类\\test.csv'
test_df=pd.read_csv(filename2)
test_df['id']=1#第一列全部替换为1
test=(test_df.values).astype(float)
for i in range(1,785):#测试集归一化
    if (np.array(test[:,i])==0).all()!=True:
        test[:,i]=Normalize(test[:,i])
arr2=np.dot(theta2,test.T)
for l in range(10000):#减去最大值防止溢出
    arr2[:,l]=arr2[:,l]-np.max(arr2[:,l],axis=0)
h_theta2=np.exp(arr2)
for i in range(0,10000):#得到对应标签的概率
    h_theta2[:,i]=softmax(h_theta2[:,i])
labelout=np.zeros(10000)
for i in range(9999):
    labelout[i]=labelfun(h_theta2[:,i+1])
labelout[9999]=labelfun(h_theta2[:,0])
print(labelout)
ids=np.arange(1,10001)
ids[9999]=0
out=pd.DataFrame({'id':ids,'label':labelout})
out.to_csv("C:\\Users\\86138\\Desktop\\submission.csv",index=0)

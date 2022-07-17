'''#torch与numpy 转换
#import torch
#import numpy as np
np_date=np.arange(6).reshape((2,3))
torch_date=torch.from_numpy(np_date)#numpy数据变成torch数据
tensor_date=torch_date.numpy()#将torch数据转换为numpy
print ("numpy",np_date)
print("torch",torch_date)
print("转换",tensor_date)
#abs绝对值
date=[-1,-2,-3,-4]
tenson_datte=torch.FloatTensor(date)
print("tenson_datte",tenson_datte)
print("torch_abs",torch.abs(tenson_datte'''


import torch.nn
'''date=[[1,2],[3,4]]
tensor=torch.FloatTensor(date)
print("numpy",np.matmul(date,date))#矩阵相乘
print("numpy_1",date.dot(date))#与上一行一样
print("torch",torch.mm(tensor,tensor))'''



'''import torch
from torch.autograd import Variable
tensor=torch.FloatTensor([[1,2],[3,4]])
variable=Variable(tensor,requires_grad=True)
t_out=torch.mean(tensor*tensor)
v_out=torch.mean(variable*variable)
v_out.backward()
print(variable.grad)
print(variable.data)
print(variable.data.numpy())'''


'''#激励函数  y=AF(Wx)
import  torch
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt
x=torch.linspace(-5,5,200)#torch.linspace把线段分为一段一段，这个代表从-5到5的距离里面划分了200个线段
x=Variable(x)
x_np=x.data.numpy()#画图时torch类型的数据不会被接受，所以改为numpy
#激励函数指令
y_relu=F.relu(x).data.numpy()
y_sigmoid=F.sigmoid(x).data.numpy()
y_tanh=F.tanh(x).data.numpy()
y_softplus=F.softplus(x).data.numpy()

plt.figure(1,figsize=(8,6))
plt.subplot(221)
plt.plot(x_np,y_relu,c='red',label='relu')
plt.ylim((-1,5))
plt.legend(loc='best')

plt.subplot(222)
plt.plot(x_np,y_sigmoid,c='red',label='sigmoid')
plt.ylim((-0.2,1.2))
plt.legend(loc='best')

plt.subplot(223)
plt.plot(x_np,y_tanh,c='red',label='tanh')
plt.ylim((-0.2,6))
plt.legend(loc='best')

plt.subplot(224)
plt.plot(x_np,y_softplus,c='red',label='softplus')
plt.ylim((-0.2,6))
plt.legend(loc='best')
plt.show()'''




'''import  torch
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt
x=torch.unsqueeze(torch.linspace(-1,1,100),dim=1)#unsqueeze数据变成二维
y=x.pow(2)+0.2*torch.rand(x.size())
x,y=Variable(x),Variable(y)
plt.ion()#变成实时过程
plt.show()
class Net(torch.nn.Module):
    def __init__(self,n_features,n_hidden,n_output):
        super(Net,self).__init__()
        self.hidden=torch.nn.Linear(n_features,n_hidden)
        self.predict=torch.nn.Linear(n_hidden,n_output)
    def forward(self,x):
        x=F.relu(self.hidden(x))
        x=self.predict(x)
        return x
net=Net(1,10,1)
print(net)
optimizer=torch .optim.SGD(net.parameters(),lr=0.5)
loss_func=torch.nn.MSELoss()#均方差
for t in range(100):
    prediction=net(x)
    loss=F.mse_loss(prediction,y)
    optimizer.zero_grad()#优化参数，把参数梯度降为0
    loss.backward()#给每一个结点计算梯度
    optimizer.step()#优化梯度
#让训练过程可视化
    if t %5==60:
        plt.cla()
        plt.scatter(x.data.numpy,y.data.numpy())
        plt.plot(x.data.numpy(),prediction.data.numpy())
        plt.text(0.5,0,'loss=%.4f' % loss.data[0],fontdict={'size':20,'color':'red'})
        plt.pause(0.1)
plt.ioff()
plt.imshow(x)
plt.show()'''



'''import  torch
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt
n_data=torch.ones(100,2)
x0=torch.normal(2*n_data,1)
y0=torch.zeros(100)
x1=torch.normal(-2*n_data,1)
y1=torch.ones(100)
x=torch.cat((x0,x1),0).type(torch.FloatTensor)#y是x的标签
y=torch.cat((y0,y1)).type(torch.LongTensor)
x,y=Variable(x),Variable(y)
plt.scatter(x.data.numpy()[:,0],x.data.numpy()[:,1],c=y.data.numpy(),s=100,lw=0)
plt.show()
class Net(torch.nn.Module):
    def __init__(self,n_features,n_hidden,n_output):
        super(Net,self).__init__()
        self.hidden=torch.nn.Linear(n_features,n_hidden)
        self.predict=torch.nn.Linear(n_hidden,n_output)
    def forward(self,x):
        x=F.relu(self.hidden(x))
        x=self.predict(x)
        return x
net=Net(2,10,1)#2个特征，10个神经元，输出2个特征
print(net)
plt.ion()
plt.show()
optimizer=torch .optim.SGD(net.parameters(),lr=0.5)
loss_func=torch.nn.MSELoss()#均方差
optimizer.zero_grad()
loss.backward()
optimizer.step()
if t%2==0:
    plt.cla()
    pediction=torch.max(F.softmax(out,dim=1),1)[1]
    pred_y=prediction.data.numpy().squeeze()
    target_y=y.data.numpy()
    plt.scatter(x.data.numpy()[:0],x.data.numpy()[:1],c=predi_y,s=100,lw=0,camp='RdYlGn')
    plt.text(1.5,-4,'Accuracy=%.2f'%accuracy,fontdict={'size':20,'color':red})
    plt.pause(0.1)
plt.ioff()
plt.show()'''



'''import  torch
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt
n_data=torch.ones(100,2)
x0=torch.normal(2*n_data,1)
y0=torch.zeros(100)
x1=torch.normal(-2*n_data,1)
y1=torch.ones(100)
x=torch.cat((x0,x1),0).type(torch.FloatTensor)#y是x的标签
y=torch.cat((y0,y1)).type(torch.LongTensor)
x,y=Variable(x),Variable(y)
plt.scatter(x.data.numpy()[:,0],x.data.numpy()[:,1],c=y.data.numpy(),s=100,lw=0)
plt.show()
net2=torch.nn.Sequential(
    torch.nn.Linear(2,10),
    torch.nn.ReLU(),
    torch.nn.Linear(10,2)
                         )
print(net2)'''
'''import torch
import torch.utils.data as Data
BATCH_SIZE=5
x=torch.linspace(1,10,100)
y=torch.linspace(10,1,100)
torch_datatset=Data.TensorDataset(data_tensor=x,target_tensor=y)
loader=Data.DataLoader(dataset=torch_datatset,
                       batch_size=BATCH_SIZE
                       shuffle=True,#打断训练顺序
                       num_workers=2
                       )
for epoch in range(3):
    for step,(batch_x,batch_y) in enumerate(loader)：'''



'''import  torch
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt
import torch.utils.data as Data
#hyper parameters
LR=0.01
BATCH_SIZE=32
EPOCH=12
x=torch.unsqueeze(torch.linspace(-1,1,100),dim=1)
y=x.pow(2)+0.1*torch.normal(torch.zeros(*x.size()))
torch_dataset=Data.TensorDataset(x,y)
loader=Data.DataLoader(dataset=torch_dataset,bath_size=BATCH_SIZE,shuffle=True,num_workers=2)
class Net(torch.nn.Module):
    def __init__(self,n_features,n_hidden,n_output):
        super(Net,self).__init__()
        self.hidden=torch.nn.Linear(1,20)
        self.predict=torch.nn.Linear(20,1)
    def forward(self,x):
        x=F.relu(self.hidden(x))
        x=self.predict(x)
        return x
net_SGD=        Net()
net_Momentnum=  Net()
net_RMSprop=    Net()
net_Adam=       Net()
nets=[net_SGD,net_Momentnum,net_RMSprop,net_Adam]
opt_SGD         =torch.optim.SGD(net_SGD.parameters(),lr=LR)
opt_Momentum    =torch.optim.SGD(net_Momentnum,parameters(),lr=LR,momentum=0.8)
opt_RMSprop     =torch.optim.RMSprop(net_RMSprop.parameters(),lr=LR,alpha=0.9)
opt_Adam        =torch.optim.Adam(net_Adam.parameters(),lr=LR,betas=(0.9,0.99))
optimizers=[opt_SGD,opt_Momentum,opt_RMSprop,opt_Adam]'''


'''import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as Data
import torchvision
import matplotlib.pyplot as plt
EPOCH=1
BATCH_SIZE=50
LR=0.001
DOWNLOAD_MNIST=False
train_data=torchvision.datasets.MNIST(
    root=',/mnist',
    train=True,
    transfrom=torchvision.transfroms.ToTensor(),
    download=DOWNLOAD_MNIST
)
print(train_data.train_data.size())
print(train_data.train_labels.size())
plt.imshow(train_data.train_data[0].numpy(),cmap='gray')
plt.title('%i'%train_data.train_labels[0])
plt.show()
train_loader=Data.Dataloader(dataset=train_data,batch_size=BATCH_SIZE，shuffle=True,num_workers=2)
test_data=torchvision.datasets.MNIST(root='./mnist/',train=False)
test_x=Variable(torch.unsqueeze(test_data,dim=1),volatile=True).type(torch.FloatTensor)[:2000]/255.
test_y=test_data.test_labels[:2000]
class CNN(nn.Moudle):
    def __init__(self):
        super(CNN,self).__init__()
        self.conv1=nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=5,
                stride=1,
                padding=2,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
self.conv2=nn.Sequential(
    nn.Conv2d(16,32,5,1,2),
    nn.ReLU(),
    nn.MaxPool2d(2)
)
self.out=nn.out=nn.Linear(32*7*7,10)
def forward(self,x):
    x=self.conv1(x)
    x=self.conv2(x)
    x=x.view(x.size(0),-1)
    output=self.out(x)
    return output
cnn=CNN()
optimizer=torch.optim.Adam(cnn.parameters(),lr=LR)
loss_func=nn.CrossEntropyLoss()
for epoch in range(EPOCH):
    for step,(x,y)in enumerate(train_loader):
        b_x=Variable(x)
        b_y=Variable(y)
        out_put=cnn(b_x)
        loss=loss_func(output,b_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()'''

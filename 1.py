# Dataset实例实战
'''from torch.utils.data import Dataset
from PIL import  Image
import os
class MyData(Dataset):
    def __init__(self,root_dir,label_dir):
        self.root_dir=root_dir
        self.label_dir=label_dir
        self.path=os.path.join(root_dir,label_dir)
        self.img_path=os.listdir(self.path)
    def __class_getitem__(self, idx):
        img_name=self.path[idx]
        img_item_path=os.path.join(self.root_dir,self.label_dir,img_name)
        img=Image.open(img_item_path)
        return img,label
    def __len__(self):
        return len(self.img_path)
root_dir="/Users/yy/PycharmProjects/pythonProject6/hymenoptera_data/train"
ants_label_dir="/Users/yy/PycharmProjects/pythonProject6/hymenoptera_data/train/ants"
ants_dataset=MyData(root_dir,ants_label_dir)'''



# tensorboard实战
import torch
import torchvision.datasets

'''from torch.utils.tensorboard import SummaryWriter
writer=SummaryWriter("logs")
from PIL import Image
import numpy as np
img_path="/Users/yy/PycharmProjects/pythonProject6/hymenoptera_data/train/ants/0013035.jpg"
img_PIL=Image.open(img_path)
img_np=np.array(img_PIL)
print(img_np.shape)
print(type(img_np))
writer.add_images("test",img_np,1,dataformats="HWC")
# for i in range(100):
#    writer.add_scalar("y=x",i,i)
writer.close()'''

#transforms实战
'''from torchvision import transforms
from PIL import Image
import cv2
from torch.utils.tensorboard import SummaryWriter
writer=SummaryWriter("logs")
img_path="/Users/yy/PycharmProjects/pythonProject6/hymenoptera_data/train/bees/16838648_415acd9e3f.jpg"
img=Image.open(img_path)
cv_img_path=cv2.imread(img_path)
tensor_trains=transforms.ToTensor()
tensor_img=tensor_trains(img)
print(tensor_img)
print(cv_img_path)
writer.add_image("tensor_img",tensor_img)
writer.close()'''


#常见的transforms
'''from torchvision import transforms
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
img_path="/Users/yy/PycharmProjects/pythonProject6/ccc.jpg"
img=Image.open(img_path)
# ToTensor
tensor_trains=transforms.ToTensor()
tensor_img=tensor_trains(img)
writer=SummaryWriter("logs")
writer.add_image("ToTensor",tensor_img)

# Normalize
print(tensor_img[0][0][0])
trans_norm=transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])#改变了像素大小
img_norm=trans_norm(tensor_img)
print(img_norm[0][0][0])
writer.add_image("Normalize",img_norm)


#resize
print(img.size)
trans_resize=transforms.Resize((512,512))
img_resize=trans_resize(img)
print(img_resize)
img_resize=tensor_trains(img_resize)
print(img_resize)
writer.add_image("Resize",img_resize,0)


#Compose
trans_resize_2=transforms.Resize(512)
print(trans_resize_2.size)
trans_compose=transforms.Compose([trans_resize_2,tensor_trains])
img_resize_2=trans_compose(img)
writer.add_image("Compose",img_resize_2,1)

#RandomCrop
trans_random=transforms.RandomCrop(512)
trans_compose_2=transforms.Compose([trans_random,trans_compose])
for i in range(10):
    img_random=trans_compose_2(img)
    writer.add_image("Random",img_random,i)
'''



#torvision数据集使用
'''
import torchvision
from torch.utils.tensorboard import SummaryWriter
data_transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
train_data=torchvision.datasets.CIFAR10("/Users/yy/PycharmProjects/pythonProject6/data",train=True,transform=data_transform,download=True)
test_data=torchvision.datasets.CIFAR10("/Users/yy/PycharmProjects/pythonProject6/data",train=False,transform=data_transform,download=True)
print(test_data[0])
print(test_data.classes)
img,target=test_data[0]
print(img)
print(target)
print(test_data.classes[target])
img.show()
writer=SummaryWriter="abc"
for i in range(10):
    img,target=test_data[i]
    writer.add_image("test_data",img,i)
writer.close()


#DataLoader的使用
import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
test_data=torchvision.datasets.CIFAR10("/Users/yy/PycharmProjects/pythonProject6/cifar-10-batches-py",train=False,
                                       transform=torchvision.transforms.ToTensor())
test_loader=DataLoader(dataset=test_data,batch_size=64,shuffle=True,num_workers=0,drop_last=True)

img_2,target_2=test_data[0]
print(img_2.shape)
print(img_2.size)
print(target)
writer=SummaryWriter("dataset")
for epoch in range(2):
    step=0
    for data in test_loader:
        img_2s,target_2s=data
        writer.add_image("datatest",img_2s,step)
        step=step+1
writer.close()
'''


# nn.moudle
'''import torch.nn as nn
import torch.nn.functional as F
class zyq(nn.Module):
    def __init__(self):
        super(zyq,self).__init__()
    def forward(self,input):
        output=input+1
        return output
ZYQ=zyq()
x=torch.tensor(1.0)
output=ZYQ(x)
print(output)'''

#卷积理解操作
'''
import torch
import torch.nn.functional as F
input=torch.tensor([[1,2,0,3,1],
                    [0,1,2,3,1],
                    [1,2,1,0,0],
                    [5,2,3,1,1],
                    [2,1,0,1,1]])

kernel=torch.tensor([[1,2,1],
                     [0,1,0],
                     [2,1,0]
                     ])
print(input.shape)
print(kernel.shape)
input=torch.reshape(input,(1,1,5,5))
kernel=torch.reshape(kernel,(1,1,3,3))
print(input.shape)
print(kernel.shape)
output=F.conv2d(input,kernel,stride=1)
print(output)
print(output.shape)
output_2=F.conv2d(input,kernel,stride=1,padding=1)
print(output_2)
print(output_2.shape)'''


#卷积层理解
'''import torch
import torchvision
from torch.utils.data import DataLoader
from torch import nn
from torch.nn import Conv2d
from torch.utils.tensorboard import SummaryWriter
dataset=torchvision.datasets.CIFAR10("/Users/yy/PycharmProjects/pythonProject6/runs",train=False,
                                     transform=torchvision.transforms.ToTensor()
                                     ,download=True)
dataloader=DataLoader(dataset,batch_size=64)
class zyq(nn.Module):
    def __init__(self):
        super(zyq,self).__init__()
        self.conv1=Conv2d(in_channels=3,out_channels=6,kernel_size=3,stride=1,padding=0)
    def forward(self,x):
        x=self.conv1(x)
        return self.conv1(x)
ZYQ=zyq()
print(ZYQ)
writer=SummaryWriter("logs")
step=0
for data in dataloader:
    imgs,targets=data
    output=ZYQ(imgs)
    print(imgs.shape)
    print(output.shape)
    writer.add_images("data",imgs,step)
    output=torch.reshape(output,(-1,3,30,30))
    writer.add_images("output",output,step)
    step = step + 1
writer.close()'''''


# 神经网络的池化层
'''
import torch
import torchvision
import torch.nn as nn
from torch.nn import MaxPool2d
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
dataloader=torchvision.datasets.CIFAR10("/Users/yy/PycharmProjects/pythonProject6/runs",train=False,
                                      transform=torchvision.transforms.ToTensor())
input=torch.tensor([[1,2,0,3,1],
                    [0,1,2,3,1],
                    [1,2,1,0,0],
                    [5,2,3,1,1],
                    [2,1,0,1,1]],dtype=torch.float32)
input=torch.reshape(input,(-1,1,5,5))
print(input.shape)
class zyq (nn.Module):
    def __init__(self):
        super(zyq, self).__init__()
        self.maxpool1=MaxPool2d(kernel_size=3,ceil_mode=False)
    def forward(self,input):
        output=self.maxpool1(input)
        return(output)
ZYQ=zyq()
writer=SummaryWriter("abc")
step=0
for data in dataloader: 
    imgs,targets=data
    writer.add_images("input",imgs,step)
    output=ZYQ(imgs)
    writer.add_images("output",output,step)
    step=step+1
writer.close()'''


#非线性激活
'''import torch
from torch import nn
from torch.nn import ReLU,Sigmoid
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
input=torch.tensor([[1,-0.5],
                    [-1,3]])
input=torch.reshape(input,(-1,1,2,2))
print(input.shape)
dataset=torchvision.datasets.CIFAR10("/Users/yy/PycharmProjects/pythonProject6/runs",train=False,download=True,
                                     transform=torchvision.transforms.ToTensor())
dataloader=DataLoader(dataset,batch_size=64)
class zyq(nn.Module):
    def __init__(self):
        super(zyq, self).__init__()
        self.relu1=ReLU()
        self.sigmoid1=Sigmoid()
    def forward(self,input):
        output=self.sigmoid1()
        return output
ZYQ=zyq()
writer=SummaryWriter("abc")
for data in dataloader:
    imgs,targets=data
    writer.add_imgaes("input",imgs,step)
    output=ZYQ(input)
    writer.add_images("output",output,step)
    step=step+1
writer.close()'''


#线性层与其他层
'''import torch
from torch.utils.data import DataLoader
from torch import nn
import torchvision
dataset=torchvision.datasets.CIFAR10("/Users/yy/PycharmProjects/pythonProject6/runs",
                       train=False,transform=torchvision.transforms.ToTensor(),
                         download=True)
dataloader=Dataloader(dataset,batch_size=64)
class zyq(nn.Module):
    def __init__(self):
        super(zyq, self).__init__()
        self.linearl=Linear(196608,10)
    def forward(self,input):
        output=self.linearl(input)
        return output
ZYQ=zyq()
for data in dataloader:
    imgs,targets=data
    print(img.shape)
    output=torch.flatten(imgs)
    print(output.shape)
    output=ZYQ(output)
    print(output.shape)'''''
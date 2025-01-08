#from torchvision import models
#from torchsummary import summary
#import torch
#import torchvision
#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#t = torchvision.models.EfficientNet().cuda
#print(t)
#summary(mobilenet_v3_small,(3,224,224))
#import torch
#import torchvision
#from torchsummary import summary          #使用 pip install torchsummary
#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#vgg = torchvision.models.RegNet().to(device)
#summary(vgg, input_size=(3, 224, 224))
#下采样从stride=(2,2)开始，通道数看下采用最后一个卷积的conv2d()中第二个值，或者BatchNorm2d（）的第一个植
#改进yolov5的backbone从后往前选出三个下采样模块
from torchvision import models
import torch
print(models.efficientnet_b0())



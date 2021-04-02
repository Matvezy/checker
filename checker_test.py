from torchvision import utils, datasets,models
import torchvision.transforms.functional as transforms
import torch.nn as nn
from PIL import Image
from torch.autograd import Variable
import torch
import cv2
from align import Aligner
class Network(nn.Module):
    def __init__(self,num_classes=3):
        super().__init__()
        self.model_name='resnet18'
        self.model=models.resnet18()
        #self.model.conv1=nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.model.fc=nn.Linear(self.model.fc.in_features, num_classes)
        
    def forward(self, x):
        x=self.model(x)
        return x

def image_loader1(image):
    """
    image = cv2.imread(image_name, 0)
    image = Image.fromarray(image)
    image = TF.resize(image, (512,512))
    """
    #image = cv2.imread(image)
    #image = Image.fromarray(image)
    #image = Image.open(image)
    image = transforms.resize(image, (312, 312))
    image = transforms.to_tensor(image)
    image = Variable(image)
    return image.cuda()
  
network = Network()
network.load_state_dict(torch.load('checker_s.pth'))
network.cuda()
aligner = Aligner(157)
image = aligner.align_image('trbl.png')     
image = image_loader1(image) 
predictions = network(image[None, ...])
m = nn.Sigmoid()
predictions = m(predictions)
print(predictions)

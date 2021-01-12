from PIL import Image
import torch
from torchvision.transforms import ToTensor
from ConvNeuralNetwork.Classes.NetModel import NetModel
import numpy as np
from PIL import Image
import torchvision.transforms as transforms


def predict(imagePath, features):
    ############ Open Image and convert to Tensor#####################
    image = Image.open(imagePath)
    image = image.resize((50, 50))
    #image = image.resize((30, 30))
    #image = ToTensor()(image).unsqueeze(0)  # unsqueeze to add artificial first dimension
    pil_to_tensor = transforms.ToTensor()(image).unsqueeze_(0)

    ############ Load (trained) NN ###################################
    net = NetModel(features)
    net.load_state_dict(torch.load('./savedCNN/Test.pth'))

    ############ Predict Feature of Image ############################
    outputs = net(pil_to_tensor)
    _, predicted = torch.max(outputs.data, 1)

    return(features[predicted.item()])

import torch
from ConvNeuralNetwork.Classes.PImagesClass import PImages
import numpy as np

from ConvNeuralNetwork.Modules.Performance import writeTestingPerformance

class Tester():
    def __init__(self, test_loader, criterion, net, plot_data, features):
        self.test_loader = test_loader
        self.criterion = criterion
        self.net = net
        self.plot_data = plot_data
        self.features = features

    def test(self):
        #device = ("cuda" if torch.cuda.is_available() else "cpu")
        device = "cpu"

        correct = 0
        total = 0

        labelList = np.array(list())
        predictedList = np.array(list())

        for i, (inputs, classes) in enumerate(self.test_loader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs = inputs.to(device)
            labels = classes.to(device)

            # forward + backward + optimize
            outputs = self.net(inputs)

            # calculate loss
            loss = self.criterion(outputs, labels).item()

            # calculate accuracy
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            labelList = np.append(labelList, labels.numpy())
            predictedList = np.append(predictedList, predicted.numpy())

        if self.plot_data:
            PImages(self.test_loader).plot()

        accuracy = 100 * correct / total

        writeTestingPerformance(loss, accuracy, labelList, predictedList, self.features)

        print('Finished Testing')
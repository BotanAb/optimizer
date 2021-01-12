import torch
import numpy as np
from torch import nn

from ConvNeuralNetwork.Modules.Performance import writeTrainingPerformance
from ConvNeuralNetwork.Modules.Performance import getAdaptedOptimizer


class Trainer():
    def __init__(self, train_loader, optimizer, criterion, net, num_epochs, demo_mode, features):
        self.train_loader = train_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.net = net
        self.num_epochs = num_epochs
        self.demo_mode = demo_mode
        self.features = features

    def train(self):
        # device = ("cuda" if torch.cuda.is_available() else "cpu")
        device = "cpu"
        print("Device = " + device)

        n = 0

        iterLoss = []
        iterAccuracy = []
        iterLabels = []
        iterPredicted = []
        iterLearningRates = []
        iterTimes = []

        iterations = 4

        for iteration in range(1, iterations + 1):

            newNet = self.net
            newNet.apply(weights_init)

            newOptimizer = self.optimizer
            adaptedOptimizer = getAdaptedOptimizer(newOptimizer, iteration)

            for g in adaptedOptimizer.param_groups:
                iterLearningRates.append(g['lr'])

            print('Iteration: ' + str(iteration))

            trainLoss = []
            trainAccuracy = []

            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)

            start.record()

            for epoch in range(self.num_epochs):  # loop over the dataset multiple times

                running_loss = 0.0
                epoch_loss = 0.0

                correct = 0
                total = 0

                labelList = np.array(list())
                predictedList = np.array(list())

                for i, (inputs, classes) in enumerate(self.train_loader, 0):

                    # get the inputs; data is a list of [inputs, labels]
                    inputs = inputs.to(device)
                    labels = classes.to(device)

                    # zero the parameter gradients
                    adaptedOptimizer.zero_grad()

                    # forward + backward + optimize
                    outputs = newNet(inputs)

                    # calculate loss
                    loss = self.criterion(outputs, labels)
                    loss.backward()
                    adaptedOptimizer.step()

                    # print loss statistics
                    running_loss += loss.item()
                    epoch_loss += loss.item()

                    # calculate accuracy
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

                    labelList = np.append(labelList, labels.numpy())
                    predictedList = np.append(predictedList, predicted.numpy())

                    n += 1
                    if n % 100 == 0:  # print every 2000 mini-batches
                        print('[%d, %5d] loss: %.3f' % (epoch + 1, n + 1, running_loss / 2))
                        if self.demo_mode:
                            log_loss(str(epoch) + ',' + str(running_loss))
                        running_loss = 0.0

                #if self.demo_mode:
                #    log_loss(str(epoch) + ',' + str(epoch_loss))

                trainLoss.append(epoch_loss)
                trainAccuracy.append(100 * correct / total)

            end.record()
            torch.cuda.synchronize()

            iterLoss.append(trainLoss)
            iterAccuracy.append(trainAccuracy)
            iterLabels.append(labelList)
            iterPredicted.append(predictedList)
            iterTimes.append(start.elapsed_time(end))

            print('Duration: ' + str(start.elapsed_time(end)))
            print('Accuracy best epoch: ' + str(100 * correct / total))

        writeTrainingPerformance(iterLoss, iterAccuracy, iterLearningRates, iterTimes, iterLabels, iterPredicted, self.features)

        print('Finished Training')


def log_loss(log_text):
    f = open('logloss.txt', 'a')
    f.write(log_text + '\n')
    f.close()


def weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform(m.weight.data)
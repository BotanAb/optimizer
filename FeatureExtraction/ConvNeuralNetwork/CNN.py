import torch.nn as nn
import torchvision.transforms as transforms

import torch

from ConvNeuralNetwork.Classes.NetModel import NetModel
from ConvNeuralNetwork.Classes.Trainer import Trainer
from ConvNeuralNetwork.Classes.Tester import Tester
from ConvNeuralNetwork.Classes.MyDataset import loadDataset

from ConvNeuralNetwork.Classes.MonitorCNN import Animate
import multiprocessing

import os

class CNN:
    def trainCNN(self, features):
        transform = transforms.Compose(
            [
                transforms.Resize((50, 50)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

        plot_data, train_CNN, demo_mode = self.adjust_workflow()
        train_loader, test_loader, classes = self.load_dataset(transform, batch_size=3, shuffle=True, pin_memory=True,
                                                               num_workers=0, features=features)
        net = self.define_cnn(features)
        criterion, optimizer = self.loss_function(net, learning_rate=0.001, momentum=0.9)

        if train_CNN:

            num_epochs = 5

            if demo_mode:
                animated_plot = Animate()
                p2 = multiprocessing.Process(target=animated_plot.start)
                p1 = multiprocessing.Process(
                    target=Trainer(train_loader, optimizer, criterion, net, num_epochs,
                            demo_mode, features).train, )

                net.train()

                p2.start()
                print("animate.start")
                p1.start()
                print("train.start")
                p1.join()
                p2.terminate()

            else:
                net.train()
                Trainer(train_loader, optimizer, criterion, net, num_epochs,
                        demo_mode, features).train()

            path = self.save_cnn(train_CNN, net)
            Tester(test_loader, criterion, net, plot_data, features).test()
            self.restore_cnn(path, features)

    def adjust_workflow(self):
        plot_data = False
        train_CNN = True
        demo_mode = False
        return plot_data, train_CNN, demo_mode

    def load_dataset(self, transform, batch_size, shuffle, pin_memory, num_workers, features):
        train_loader, test_loader, classes = loadDataset(transform, batch_size, shuffle, pin_memory, num_workers,
                                                         features)
        return train_loader, test_loader, classes

    def define_cnn(self, features):
        net = NetModel(features)
        return net

    def loss_function(self, net, learning_rate, momentum):
        import torch.optim as optim
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=momentum)
        return criterion, optimizer

    def save_cnn(self, train_CNN, net):
        cnn_path = './savedCNN/'
        cnn_name = 'Test.pth'

        if not os.path.exists(cnn_path):
            os.makedirs(cnn_path)

        if train_CNN:
            torch.save(net.state_dict(), cnn_path + cnn_name)
        return cnn_path + cnn_name

    def restore_cnn(self, PATH, features):
        net = NetModel(features)
        net.load_state_dict(torch.load(PATH))


if __name__ == '__main__':
    features = ["fist", "palm", "thumb"]
    test = CNN()
    test.trainCNN(features)

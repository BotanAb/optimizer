import torch.nn as nn
import torch.nn.functional as F
import torch
import matplotlib.pyplot as plt

class NetModel(nn.Module):
	def __init__(self, features):
		super(NetModel, self).__init__()
		self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5)
		self.pool = nn.MaxPool2d(2, 2)
		self.conv2 = nn.Conv2d(6, 16, 5)
		self.fc1 = nn.Linear(16 * 9 * 9, 120)
		self.fc2 = nn.Linear(120, 84)
		self.fc3 = nn.Linear(84, len(features))  # outputs ( _, amount of outputs)

	def forward(self, x):
		x = self.pool(F.relu(self.conv1(x)))
		x = self.pool(F.relu(self.conv2(x)))
		#print(x.size())
		x = x.view(-1, 16 * 9 * 9)
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		x = self.fc3(x)
		return x


#################################################################################
#
# Test
#
#################################################################################
if __name__ == "__main__":
	test = NetModel()
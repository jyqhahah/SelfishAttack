import torch
import torch.nn as nn

class CNNCifar(nn.Module):
    def __init__(self, in_channel, num_classes):
        super(CNNCifar, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, 30, 3)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(30, 50, 3)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.linear1 = nn.Linear(1800, 100)
        self.linear2 = nn.Linear(100, num_classes)
        self._init_weight_()

    def forward(self, x):
        x = self.conv1(x)
        x = torch.relu(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = torch.relu(x)
        x = self.maxpool2(x)
        x = x.view(x.size(0), -1)
        x = self.linear1(x)
        x = torch.relu(x)
        x = self.linear2(x)
        return x

    def _init_weight_(self):
        nn.init.normal_(self.linear1.weight, std=0.01)
        nn.init.normal_(self.linear2.weight, std=0.01)

    def get_all_gradient(self):
        para_list = []
        for param in self.parameters():
            para_list.append(param.grad.clone())
        return para_list

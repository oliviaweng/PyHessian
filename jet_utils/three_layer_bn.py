import torch.nn as nn


class ThreeLayer(nn.Module):
    def __init__(self):
        super(ThreeLayer, self).__init__()

        self.dense_1 = nn.Linear(16, 64)
        self.dense_2 = nn.Linear(64, 32)
        self.dense_3 = nn.Linear(32, 32)
        self.dense_4 = nn.Linear(32, 5)

        self.act = nn.ReLU()
        self.softmax = nn.Softmax(1)

    def forward(self, x):
        relu_1 = self.act(self.dense_1(x))
        relu_1.retain_grad()
        relu_2 = self.act(self.dense_2(relu_1))
        relu_2.retain_grad()
        relu_3 = self.act(self.dense_3(relu_2))
        relu_3.retain_grad()
        return self.softmax(self.dense_4(relu_3))


class ThreeLayerBN(nn.Module):
    def __init__(self):
        super(ThreeLayerBN, self).__init__()

        self.dense_1 = nn.Linear(16, 64)
        self.dense_2 = nn.Linear(64, 32)
        self.dense_3 = nn.Linear(32, 32)
        self.dense_4 = nn.Linear(32, 5)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(32)
        self.bn3 = nn.BatchNorm1d(32)

        self.act = nn.ReLU()
        self.softmax = nn.Softmax(1)

    def forward(self, x):
        relu_1 = self.act(self.bn1(self.dense_1(x)))
        relu_1.retain_grad()
        relu_2 = self.act(self.bn2(self.dense_2(relu_1)))
        relu_2.retain_grad()
        relu_3 = self.act(self.bn3(self.dense_3(relu_2)))
        relu_3.retain_grad()
        return self.softmax(self.dense_4(relu_3)), relu_1, relu_2, relu_3


def three_layer():
    return ThreeLayer()


def three_layer_bn():
    return ThreeLayerBN()

import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.Maxout import Maxout

"""
Simple implementation of Uni-modal ConditionalGAN introduced by Mirza et al. 2014.
"""


class Generator(nn.Module):
    def __init__(self):
        super().__init__()

        # Layers
        self.linear1_noise = nn.Linear(100, 200)  # Mapping noise 100 to 200
        self.linear1_labels = nn.Linear(10, 1000)  # Mapping label 10 to 1000
        self.linear2 = nn.Linear(200 + 1000, 1200)  # Mapping Noise layer and Label linear into 1200
        self.linear3 = nn.Linear(1200, 784)  # Linear for final result
        self.dropout = nn.Dropout(0)

        # Activation Layers
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, noise, labels):
        nl1 = self.relu(self.linear1_noise(noise))  # Liner layer of Noise
        nl1 = self.dropout(nl1)
        ll1 = self.relu(self.linear1_labels(
            F.one_hot(labels, num_classes=10).type(torch.FloatTensor).to('cuda')))  # Liner layer of Label
        ll1 = self.dropout(ll1)
        l2 = self.relu(self.linear2(torch.cat((nl1, ll1), 1)))  # Mapping both layer
        l2 = self.dropout(l2)
        l3 = self.sigmoid(self.linear3(l2))  # Final linear
        return l3


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()

        # Layers
        self.maxout1_x = Maxout(in_dim=784, num_piece=5, num_units=240)
        self.maxout1_y = Maxout(in_dim=10, num_piece=5, num_units=50, is_label=True, num_class=10)
        self.maxout2 = Maxout(in_dim=290, num_piece=4, num_units=240)
        self.linear1 = nn.Linear(240, 1)
        self.dropout = nn.Dropout(0)

        # Activation Layers
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, images, labels):
        m1x = self.maxout1_x(images)
        m1x = self.dropout(m1x)
        m1y = self.maxout1_y(labels)
        m1y = self.dropout(m1y)
        m2 = self.maxout2(torch.cat((m1x, m1y), 1))
        m2 = self.dropout(m2)
        sl = self.sigmoid(self.linear1(m2))
        return sl

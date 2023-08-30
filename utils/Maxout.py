import torch
import torch.nn as nn
import torch.nn.functional as F


class Maxout(nn.Module):
    """
    simple implements Maxout unit introduced by Goodfellow et al. 2013 for cGAN by Mirza et al. 2014.

    :param in_dim: input size
    :param num_units: number of unit
    :param num_piece: output size
    :param is_label: is input type is label
    :param num_class: for one-hot encoding
    """

    def __init__(self, in_dim, num_units, num_piece, is_label: bool = False, num_class=0):
        super().__init__()

        # Parameters
        self.in_dim = in_dim
        self.num_units = num_units
        self.num_piece = num_piece
        self.num_class = num_class
        self.is_label = is_label

        # Layer
        self.linear1 = nn.Linear(in_dim, num_units * num_piece)

    def forward(self, x):
        # Convert label to one-hot vector
        if self.is_label:
            x = F.one_hot(x, num_classes=self.num_class).type(torch.FloatTensor).to('cuda')

        x = self.linear1(x)
        x = torch.max(x.view(-1, self.num_units, self.num_piece), dim=2).values
        return x

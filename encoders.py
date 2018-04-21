import numpy as np
from torch import nn


class Encoder(nn.Module):
    def __init__(self, args):
        super(Encoder, self).__init__()
        self.shape = (1, 3, 3)
        self.dim = 3
        convblock = nn.Sequential(
                nn.Conv2d(1, self.dim, 1, stride=2, padding=1),
                nn.Dropout(p=0.3),
                nn.ReLU(True),
                nn.Conv2d(self.dim, 2*self.dim, 1, stride=1, padding=2),
                nn.Dropout(p=0.3),
                nn.ReLU(True),
                nn.Conv2d(2*self.dim, 4*self.dim, 1, stride=1, padding=2),
                nn.Dropout(p=0.3),
                nn.ReLU(True),
                )
        self.main = convblock
        self.output = nn.Linear(4*4*4*self.dim, self.dim)

    def forward(self, input):
        input = input.view(-1, 1, 3, 3)
        out = self.main(input)
        out = out.view(-1, 4*4*4*self.dim)
        out = self.output(out)
        return out.view(-1, self.dim)

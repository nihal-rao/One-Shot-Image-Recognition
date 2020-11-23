import torch

from torch import nn

class Siamese(nn.Module):
    def __init__(self):
        super(Siamese, self).__init__()
        self.conv = nn.Sequential(nn.Conv2d(1, 64, 10),nn.ReLU(),
        nn.MaxPool2d(2),nn.Conv2d(64, 128, 7),nn.ReLU(),nn.MaxPool2d(2),
        nn.Conv2d(128, 128, 4),nn.ReLU(),nn.MaxPool2d(2),nn.Conv2d(128, 256, 4),
        )

        self.fcn_1= nn.Sequential(nn.Linear(9216, 4096), nn.Sigmoid())
        self.fcn_2 = nn.Sequential(nn.Linear(4096,1), nn.Sigmoid())

    def forward(self, img_1, img_2):
        f_1 = self.conv(img_1)
        f_1 = f_1.view(f_1.shape[0], -1)
        f_1 = self.fcn_1(f_1)

        f_2 = self.conv(img_2)
        f_2 = f_2.view(f_2.shape[0], -1)
        f_2 = self.fcn_1(f_2)

        result = self.fcn_2(torch.abs(f_1-f_2))

        return result
import torch.nn as nn

class Dip_Cnn(nn.Module):
    def __init__(self):
        super(Dip_Cnn, self).__init__()
        self.cnnnet = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1, bias=True),
            nn.LeakyReLU(negative_slope=0.1),

            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1, bias=True),
            nn.LeakyReLU(negative_slope=0.1),

            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1, bias=True),
            nn.LeakyReLU(negative_slope=0.1),

            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1, bias=True),
            nn.LeakyReLU(negative_slope=0.1),

            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1, bias=True),
            nn.LeakyReLU(negative_slope=0.1),
        )
        self.full_layers1 = nn.Sequential(
            nn.Linear(2048, 64),  # Adjusted the number of input features to 2048
            nn.Linear(64, 15),
        )

    def forward(self, x):
        out = self.cnnnet(x)
        out = out.view(out.size(0), -1)
        out = self.full_layers1(out)
        return out
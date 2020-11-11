from torch import nn


class LatentNetwork(nn.Module):
    def __init__(self, input_nc = 1, filter_size = 8):
        super(LatentNetwork, self).__init__()

        #input Z
        model = [nn.ConvTranspose2d(in_channels = input_nc, out_channels = filter_size, kernel_size = 4, stride = 2, padding = 1),
                 nn.BatchNorm2d(filter_size),
                 nn.ReLU(inplace=True)]

        in_features = filter_size
        out_features = in_features * 2
        for i in range(2):
            model += [nn.ConvTranspose2d(in_features, out_features, 4, stride=2, padding=1),
                      nn.InstanceNorm2d(out_features),
                      nn.ReLU(inplace=True)]
            in_features = out_features
            out_features = in_features * 2

        model += [nn.ConvTranspose2d(in_features, input_nc, 4, stride=2, padding=1),
                  nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


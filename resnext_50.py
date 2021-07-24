import torchvision.models as models
import torch.nn as nn
import torch


class Resnext_50(nn.Module):
    def __init__(self, num_class=10):
        super(Resnext_50, self).__init__()
        self.num_class = num_class
        self.feature = nn.Sequential(*list(models.resnext50_32x4d(pretrained=True).children())[:-1])
        self.fc = nn.Linear(2048 * 1 * 1, self.num_class)

    def forward(self, x):
        x = self.feature(x)
        batchsz = x.size(0)
        x = x.view(batchsz, 2048)
        x = self.fc(x)
        return x


if __name__ == '__main__':
    x = torch.randn(4, 3, 128, 128)
    model = Resnext_50()
    # print(model)
    out = model(x)
    print(out.shape)
import torch
import torch.nn as nn

class ResBlockResnet8(nn.Module):
    def __init__(self, in_channel, out_channel) -> None:
        super(ResBlockResnet8, self).__init__()
        self.batch1 = nn.BatchNorm2d(in_channel)
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=2, padding=1)

        self.batch2 = nn.BatchNorm2d(out_channel)
        self.conv2 = nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1)

        self.conv_out = nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=2, padding=0)
    def forward(self, x):
        f = self.batch1(x)
        f = self.relu(f)
        f = self.conv1(f)

        f = self.batch2(f)
        f = self.relu(f)
        f = self.conv2(f)
        
        x = self.conv_out(x)
        return x + f

class Resnet8(nn.Module):
    def __init__(self):
        super(Resnet8, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, (5,5), stride=2, padding=2)
        self.pool1 = nn.MaxPool2d(3 , 2)

        self.res1 = ResBlockResnet8(32, 32)
        self.res2 = ResBlockResnet8(32, 64)
        self.res3 = ResBlockResnet8(64, 128)

        self.fc1 = nn.Linear(1536, 512)
        self.fc2 = nn.Linear(512, 4)

        self.dropout = nn.Dropout(0.5)
        self.af = nn.Softmax()
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        x = torch.flatten(x, 1)
        
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        x = self.af(x)
        return x
        


model = Resnet8()
params = sum(p.numel() for p in model.parameters())
print("Number of parameters:", params)


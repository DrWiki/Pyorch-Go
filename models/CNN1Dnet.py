import torch
import torch.nn as nn
class Conv1DNet(nn.Module):
    def __init__(self, num_classes):
        super(Conv1DNet, self).__init__()
        self.conv1 = nn.Conv1d(1, 32, kernel_size=15, stride=2, padding=7, bias=False)
        self.bn1 = nn.BatchNorm1d(32)
        self.relu1 = nn.ReLU(inplace=True)
        self.maxpool1 = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        # self.dp1 = nn.Dropout(p=0.2)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=15, stride=2, padding=7, bias=False)
        self.bn2 = nn.BatchNorm1d(64)
        self.relu2 = nn.ReLU(inplace=True)
        self.maxpool2 = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        # self.dp2 = nn.Dropout(p=0.2)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=15, stride=2, padding=7, bias=False)
        self.bn3 = nn.BatchNorm1d(128)
        self.relu3 = nn.ReLU(inplace=True)
        self.maxpool3 = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        # self.dp3 = nn.Dropout(p=0.1)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.maxpool1(x)
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.maxpool2(x)
        x = self.relu3(self.bn3(self.conv3(x)))
        x = self.maxpool3(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class Conv1DNetMAX(nn.Module):
    def __init__(self, num_classes):
        super(Conv1DNetMAX, self).__init__()
        self.conv1 = nn.Conv1d(1, 256, kernel_size=15, stride=2, padding=7, bias=False)
        self.bn1 = nn.BatchNorm1d(256)
        self.relu1 = nn.ReLU(inplace=True)
        self.maxpool1 = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        # self.dp1 = nn.Dropout(p=0.2)
        self.conv2 = nn.Conv1d(256, 512, kernel_size=15, stride=2, padding=7, bias=False)
        self.bn2 = nn.BatchNorm1d(512)
        self.relu2 = nn.ReLU(inplace=True)
        self.maxpool2 = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        # self.dp2 = nn.Dropout(p=0.2)
        self.conv3 = nn.Conv1d(512, 1024, kernel_size=15, stride=2, padding=7, bias=False)
        self.bn3 = nn.BatchNorm1d(1024)
        self.relu3 = nn.ReLU(inplace=True)
        self.maxpool3 = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        # self.dp3 = nn.Dropout(p=0.1)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.maxpool1(x)
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.maxpool2(x)
        x = self.relu3(self.bn3(self.conv3(x)))
        x = self.maxpool3(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
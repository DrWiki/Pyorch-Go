import torch
import torch.nn as nn

class Model_Fit(nn.Module):
    def __init__(self,N = 16):
        super(Model_Fit, self).__init__()
        self.fc1 = nn.Linear(16, 500)
        self.R1 = nn.ReLU()
        self.fc2 = nn.Linear(500, 1000)
        self.R2 = nn.ReLU()
        self.fc3 = nn.Linear(1000, N)

    def forward(self, x):
        x = self.fc1(x)
        x = self.R1(x)
        x = self.fc2(x)
        x = self.R2(x)
        x = self.fc3(x)
        return x
    def name(self):
        return self.__class__.__name__


class Model_Fit_s(nn.Module):
    def __init__(self,N=16):
        super(Model_Fit_s, self).__init__()
        self.fc1 = nn.Linear(16, 30)
        self.R1 = nn.ReLU()
        self.fc2 = nn.Linear(30, 100)
        self.R2 = nn.ReLU()
        self.fc3 = nn.Linear(100, N)

    def forward(self, x):
        x = self.fc1(x)
        x = self.R1(x)
        x = self.fc2(x)
        x = self.R2(x)
        x = self.fc3(x)
        return x
    def name(self):
        return self.__class__.__name__

class Model_Fit_ss(nn.Module):
    def __init__(self, N):
        super(Model_Fit_ss, self).__init__()
        self.fc1 = nn.Linear(16, 16)
        self.R1 = nn.ReLU()
        self.fc2 = nn.Linear(16, 16)
        self.R2 = nn.ReLU()
        self.fc3 = nn.Linear(16, N)

    def forward(self, x):
        x = self.fc1(x)
        x = self.R1(x)
        x = self.fc2(x)
        x = self.R2(x)
        x = self.fc3(x)
        return x
    def name(self):
        return self.__class__.__name__

class Model_Fit_sss(nn.Module):
    def __init__(self, N):
        super(Model_Fit_sss, self).__init__()
        self.fc1 = nn.Linear(16, 8)
        self.R1 = nn.ReLU()
        self.fc2 = nn.Linear(8, 8)
        self.R2 = nn.ReLU()
        self.fc3 = nn.Linear(8, N)

    def forward(self, x):
        x = self.fc1(x)
        x = self.R1(x)
        x = self.fc2(x)
        x = self.R2(x)
        x = self.fc3(x)
        return x
    def name(self):
        return self.__class__.__name__
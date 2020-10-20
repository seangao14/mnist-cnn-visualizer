import torch.nn as nn
import torch.nn.functional as F

'''
since this model is purely for visualization purposes, performance is not of utmost importance
with only this amount of features, the model is able to achieve 92% accuracy
'''

class cnn(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv2d(1, 2, 5)
        self.c2 = nn.Conv2d(2, 2, 5)
        self.l1 = nn.Linear(2*4*4, 10)
    
    def forward(self, x):
        x = F.max_pool2d(F.relu(self.c1(x)), (2,2))
        x = F.max_pool2d(F.relu(self.c2(x)), (2,2))
        # print(x.shape)
        x = x.view(-1, 2*4*4)
        x = self.l1(x)
        return x
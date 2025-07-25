import torch
import torch.nn as nn 
import torch.nn.functional as F
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import torch.optim as optim

class Model(nn.Module):
    def __init__(self,inputs,layer1,layer2,outputs):
        super(Model, self).__init__()
        self.fc = nn.Linear(inputs,layer1)
        self.fc2 = nn.Linear(layer1,layer2)
        self.fc3 = nn.Linear(layer2,outputs)
    def forward(self, x):
        x = F.relu(self.fc(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)
    
class DigitsClassifier(nn.Module):
    def __init__(self,):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1,out_channels=16,kernel_size=3)
        self.max1 = nn.MaxPool2d(kernel_size=2,stride=2)
        self.conv2 = nn.Conv2d(in_channels=16,out_channels=16,kernel_size=3)
        self.max2 = nn.MaxPool2d(kernel_size=3,stride=2)
        self.conv3 = nn.Conv2d(in_channels=16,out_channels=10,kernel_size=3)
        self.fc1 = nn.Linear(10*3*3,288)
        self.fc2 = nn.Linear(288,64)
        self.output = nn.Linear(64,10)

    def forward(self,x):
        x = F.relu(self.conv1(x))
        x = self.max1(x)
        x = F.relu(self.conv2(x))
        x = self.max2(x)
        x = F.relu(self.conv3(x))
        x = x.view(-1,10*3*3)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.softmax(self.output(x),dim=1)
        return x


model = DigitsClassifier()
optimizer = optim.Adam(model.parameters(),lr=0.01)  
model.load_state_dict(torch.load('model.pth',map_location=torch.device('cpu')))
test_data = pd.read_csv("C:/Users/Best/OneDrive/Documents/deep learning/MNIST_CSV/Mnist_test.csv")
test_data = test_data.to_numpy()
test_x = test_data[:,1:].reshape((9999,1,28,28))
test_y = test_data[:,:1].reshape((9999))
samplex = torch.tensor(test_x[3],dtype=torch.float32)
sampley = torch.tensor(test_y[3],dtype=torch.long)
result = model(samplex)

samplex = np.array(samplex.squeeze(0),copy=True,dtype=np.float32)
def findmax(x):
    x = list(x.squeeze(0).detach().numpy())
    max = x[0]
    index = 0
    for i in range(len(x)):
        if x[i] > max:
            max = x[i]
            index = i
    return index
print(f"Model Output : {findmax(result)} Actual Output : {sampley}")
plt.imshow(samplex,cmap='gray')
plt.title("The Image")
plt.show()
print(samplex.shape)

#print(model.forward(torch.tensor([6.4,2.8,5.6,2.1],dtype=torch.float32)))

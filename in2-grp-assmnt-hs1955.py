# %%
# Very Helpful link
# https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html

# %%
# Makes File Handling Easier
import os

# PyTorch model and training necessities
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# from torch.utils.data import DataLoader

# Image datasets and image manipulation
import torchvision
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.io import read_image
import torchvision.transforms as transforms
from PIL import Image
import pandas as pd

# Image display
import matplotlib.pyplot as plt
import numpy as np

# PyTorch TensorBoard support
from torch.utils.tensorboard.writer import SummaryWriter

# ? Not sure what these are for
import scipy.io as scio
import shutil


# %%
# Create train, valid and test directories to sort dataset into.
def makePartitionDirs():
    for i in range(1, 103):
        os.makedirs("data/102flowers/train/" + str(i), exist_ok=True)
        os.makedirs("data/102flowers/test/" + str(i), exist_ok=True)
        os.makedirs("data/102flowers/valid/" + str(i), exist_ok=True)


# Distribute dataset into train, valid and test directories according to setid.mat specifications.
def partitionData(imageLabels, setid, sortedPath, dataPath):
    for i in range(len(imageLabels["labels"][0])):
        filename = "image_" + str(i + 1).zfill(5) + ".jpg"
        if i + 1 in setid["trnid"][0]:
            targetFolder = os.path.join(
                sortedPath, "train", str(imageLabels["labels"][0][i])
            )
        elif i + 1 in setid["valid"][0]:
            targetFolder = os.path.join(
                sortedPath, "valid", str(imageLabels["labels"][0][i])
            )
        else:
            targetFolder = os.path.join(
                sortedPath, "test", str(imageLabels["labels"][0][i])
            )
        shutil.copy(
            os.path.join(dataPath, filename), os.path.join(targetFolder, filename)
        )


# %%
trainTransforms = transforms.Compose(
    [
        transforms.Resize(160),
        transforms.RandomRotation(90),
        transforms.CenterCrop(128),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)
testTransforms = validTransforms = transforms.Compose(
    [
        transforms.Resize(160),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)

# %%
dataPath = "data/102flowers/jpg"
sortedPath = "data/102flowers"
setid = scio.loadmat(f"data/setid.mat")
imageLabels: dict = scio.loadmat(f"data/imagelabels.mat")
# Call these if you don't have the directories set up as needed.
makePartitionDirs()
partitionData(imageLabels, setid, sortedPath, dataPath)
trainingData = datasets.ImageFolder(
    root="data/102flowers/train", transform=trainTransforms
)
validationData = datasets.ImageFolder(
    root="data/102flowers/valid", transform=validTransforms
)
testingData = datasets.ImageFolder(
    root="data/102flowers/test", transform=testTransforms
)

# %%
# Data loaders for use as input.
in_batch_size = 16
trainDataLoader = torch.utils.data.DataLoader(
    trainingData, batch_size=in_batch_size, shuffle=True
)
validDataLoader = torch.utils.data.DataLoader(
    validationData, batch_size=in_batch_size, shuffle=False
)
testDataLoader = torch.utils.data.DataLoader(
    testingData, batch_size=in_batch_size, shuffle=False
)


# %%
def showImage(image):
    npImage = image.numpy()
    plt.imshow(np.transpose(npImage, (1, 2, 0)))
    plt.show()


torch.manual_seed(12345)
dataIter = iter(trainDataLoader)
images, labels = next(dataIter)
print(len(images))
print(labels)
showImage(torchvision.utils.make_grid(images))
print(" ".join(f"{labels[j]}" for j in range(4)))

# %%
figure = plt.figure(figsize=(8, 8))
cols, rows = 3, 3
for i in range(1, cols * rows + 1):
    sample_idx = torch.randint(len(trainingData), size=(1,)).item()
    img, label = trainingData[int(sample_idx)]
    figure.add_subplot(rows, cols, i)
    plt.title(label)
    plt.axis("off")
    print(img.squeeze().shape)
    processed_image = np.rollaxis(img.squeeze().numpy(), 0, 3)
    plt.imshow(processed_image)
plt.show()

# %%
# # The CNN Network
# class Net(nn.Module):
#     def __init__(self, img_size=256):
#         super(Net, self).__init__()
#         self.img_size = img_size
#         # 3 input image channel, 6 output channels, 5x5 square convolution
#         # kernel
#         # self.conv1 = nn.Conv2d(3, 6, img_size)
#         # self.conv2 = nn.Conv2d(6, 16, img_size)
#         # # an affine operation: y = Wx + b
#         # self.fc1 = nn.Linear(16 * img_size**2, 120)  # 5*5 from image dimension
#         # self.fc2 = nn.Linear(120, 84)
#         # self.fc3 = nn.Linear(84, 10)

#         conv1_1 =   nn.Conv2d()          #  64 3x3x3 convolutions with stride [1  1] and padding [1  1  1  1]
#         relu1_1 =   nn.ReLU()            #       ReLU
#         conv1_2 =   nn.Conv2d()          #  64 3x3x64 convolutions with stride [1  1] and padding [1  1  1  1]
#         relu1_2 =   nn.ReLU()            #       ReLU
#         pool1 =     nn.MaxPool2d()        #    2x2 max pooling with stride [2  2] and padding [0  0  0  0]
#         conv2_1 =   nn.Conv2d()          #  128 3x3x64 convolutions with stride [1  1] and padding [1  1  1  1]
#         relu2_1 =   nn.ReLU()            #       ReLU
#         conv2_2 =   nn.Conv2d()          #  128 3x3x128 convolutions with stride [1  1] and padding [1  1  1  1]
#         relu2_2 =  nn.ReLU()             #      ReLU
#         pool2 =    nn.MaxPool2d()         #   2x2 max pooling with stride [2  2] and padding [0  0  0  0]
#         conv3_1 =  nn.Conv2d()           # 256 3x3x128 convolutions with stride [1  1] and padding [1  1  1  1]
#         relu3_1 =  nn.ReLU()             #      ReLU
#         conv3_2 =  nn.Conv2d()           # 256 3x3x256 convolutions with stride [1  1] and padding [1  1  1  1]
#         relu3_2 =  nn.ReLU()             #      ReLU
#         conv3_3 =  nn.Conv2d()           # 256 3x3x256 convolutions with stride [1  1] and padding [1  1  1  1]
#         relu3_3 =  nn.ReLU()             #      ReLU
#         conv3_4 =  nn.Conv2d()           # 256 3x3x256 convolutions with stride [1  1] and padding [1  1  1  1]
#         relu3_4 =  nn.ReLU()             #      ReLU
#         pool3 =    nn.MaxPool2d()         #   2x2 max pooling with stride [2  2] and padding [0  0  0  0]
#         conv4_1 =  nn.Conv2d()           # 512 3x3x256 convolutions with stride [1  1] and padding [1  1  1  1]
#         relu4_1 =  nn.ReLU()             #      ReLU
#         conv4_2 =  nn.Conv2d()           # 512 3x3x512 convolutions with stride [1  1] and padding [1  1  1  1]
#         relu4_2 =  nn.ReLU()             #      ReLU
#         conv4_3 =  nn.Conv2d()           # 512 3x3x512 convolutions with stride [1  1] and padding [1  1  1  1]
#         relu4_3 =  nn.ReLU()             #      ReLU
#         conv4_4 =  nn.Conv2d()           # 512 3x3x512 convolutions with stride [1  1] and padding [1  1  1  1]
#         relu4_4 =  nn.ReLU()             #      ReLU
#         pool4 =    nn.MaxPool2d()         #   2x2 max pooling with stride [2  2] and padding [0  0  0  0]
#         conv5_1 =  nn.Conv2d()           # 512 3x3x512 convolutions with stride [1  1] and padding [1  1  1  1]
#         relu5_1 =  nn.ReLU()             #      ReLU
#         conv5_2 =  nn.Conv2d()           # 512 3x3x512 convolutions with stride [1  1] and padding [1  1  1  1]
#         relu5_2 =  nn.ReLU()             #      ReLU
#         conv5_3 =  nn.Conv2d()           # 512 3x3x512 convolutions with stride [1  1] and padding [1  1  1  1]
#         relu5_3 =  nn.ReLU()             #      ReLU
#         conv5_4 =  nn.Conv2d()           # 512 3x3x512 convolutions with stride [1  1] and padding [1  1  1  1]
#         relu5_4 =  nn.ReLU()             #      ReLU
#         pool5 =    nn.MaxPool2d()         #   2x2 max pooling with stride [2  2] and padding [0  0  0  0]
#         fc6 =      nn.Linear()     #   4096 fully connected layer
#         relu6 =    nn.ReLU()             #      ReLU
#         drop6 =    nn.Dropout2d()             #   50% dropout
#         fc7 =      nn.Linear()     #   4096 fully connected layer
#         relu7 =    nn.ReLU()             #      ReLU
#         drop7 =    nn.Dropout2d()             #   50% dropout
#         fc8 =      nn.Linear()     #   1000 fully connected layer
#         prob =     nn.Softmax2d()              #   softmax
#         # output =   # Classification Output #   crossentropyex with 'tench' and 999 other classes

#     def forward(self, x):
#         # # Max pooling over a (2, 2) window
#         # x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
#         # If the size is a square, you can specify with a single number
#         # x = F.max_pool2d(F.relu(self.conv2(x)), self.img_size)
#         # x = torch.flatten(x, 1)  # flatten all dimensions except the batch dimension
#         # x = F.relu(self.fc1(x))
#         # x = F.relu(self.fc2(x))
#         # x = self.fc3(x)
#         # return x
#         x =
#         return x


# net = Net()
# net


# %%
# The CNN Network
# Define a convolution neural network
class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()

        self.conv1 = nn.Conv2d(
            in_channels=3, out_channels=12, kernel_size=5, stride=1, padding=1
        )
        self.bn1 = nn.BatchNorm2d(12)
        self.conv2 = nn.Conv2d(
            in_channels=12, out_channels=12, kernel_size=5, stride=1, padding=1
        )
        self.bn2 = nn.BatchNorm2d(12)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv4 = nn.Conv2d(
            in_channels=12, out_channels=24, kernel_size=5, stride=1, padding=1
        )
        self.bn4 = nn.BatchNorm2d(24)
        self.conv5 = nn.Conv2d(
            in_channels=24, out_channels=24, kernel_size=5, stride=1, padding=1
        )
        self.bn5 = nn.BatchNorm2d(24)
        self.fc1 = nn.Linear(24 * 10 * 10, 10)

    def forward(self, input):
        output = F.relu(self.bn1(self.conv1(input)))
        output = F.relu(self.bn2(self.conv2(output)))
        output = self.pool(output)
        output = F.relu(self.bn4(self.conv4(output)))
        output = F.relu(self.bn5(self.conv5(output)))
        output = output.view(-1, 24 * 10 * 10)
        output = self.fc1(output)

        return output


# Instantiate a neural network model
model = Network()

# %%
# See the learnable parameters of our model
params = list(net.parameters())
print(len(params))
print(params[0].size())  # conv1's .weight

# %%
# Generate a random 32x32 image, what the models wants
input = torch.randn(1, 1, 32, 32)
out = net(input)
print(out)

# %%
# Zero the gradient buffers of all parameters and backprops with random gradients
net.zero_grad()
out.backward(torch.randn(1, 10))

# %%
# Does this device have a GPU?
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Assuming that we are on a CUDA machine, this should print a CUDA device:
print(device)

# Use the GPU if its there, otherwise use the CPU
net.to(device)

# %%
# Loss function
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

# %%
# Train the Network
for epoch in range(2):  # loop over the dataset multiple times
    running_loss = 0.0
    for i, data in enumerate(trainDataLoader, 0):
        # Get the inputs; data is a list of [inputs, labels] # inputs, labels = data
        # And send all the inputs and targets at every step to the chosen device
        inputs, labels = data[0].to(device), data[1].to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)  # This is the Forward Pass
        loss = criterion(outputs, labels)
        loss.backward()  # This is the Backward Pass
        optimizer.step()  # This is the optimizer

        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:  # print every 2000 mini-batches
            print(f"[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}")
            running_loss = 0.0

print("Finished Training")

# %%
# Save the trained network
PATH = "./flowers102.pth"
torch.save(net.state_dict(), PATH)

# %%
# Test the network
dataiter = iter(testDataLoader)
images, labels = next(dataiter)

# print images
plt.imshow(torchvision.utils.make_grid(images))
print("GroundTruth: ", " ".join(f"{labels[j]:5s}" for j in range(4)))

# %%
# Load back the saved network
net = Net()
net.load_state_dict(torch.load(PATH))

# %%
# Get predictions
outputs = net(images)

# %%
# Lets get the images which the AI thinks is the strongest case of each class
_, predicted = torch.max(outputs, 1)

print("Predicted: ", " ".join(f"{predicted[j]:5s}" for j in range(4)))

# %%
# Gauge the performance of the network
correct = 0
total = 0
# since we're not training, we don't need to calculate the gradients for our outputs
with torch.no_grad():
    for data in testDataLoader:
        images, labels = data
        # calculate outputs by running images through the network
        outputs = net(images)
        # the class with the highest energy is what we choose as prediction
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Accuracy of the network on the 10000 test images: {100 * correct // total} %")

# %%
# Get the predictions accuracy of each class

# prepare to count predictions for each class
correct_pred = {classname: 0 for classname in labels}
total_pred = {classname: 0 for classname in labels}

# again no gradients needed
with torch.no_grad():
    for data in testDataLoader:
        images, labels = data
        outputs = net(images)
        _, predictions = torch.max(outputs, 1)
        # collect the correct predictions for each class
        for label, prediction in zip(labels, predictions):
            if label == prediction:
                correct_pred[label] += 1
            total_pred[label] += 1


# print accuracy for each class
for classname, correct_count in correct_pred.items():
    accuracy = 100 * float(correct_count) / total_pred[classname]
    print(f"Accuracy for class: {classname:5s} is {accuracy:.1f} %")

# %%
# Cleanup
del dataiter

# %%

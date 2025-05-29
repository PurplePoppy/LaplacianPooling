import torch.optim as optim
from torchvision import datasets
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np
import os

setup_seed(4)

class CNNModel(nn.Module):
    def __init__(self, input_size, output_size, k):
        super(CNNModel, self).__init__()
        self.conv_layer = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=18, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=18, out_channels=36, kernel_size=k, stride=k),
            nn.ReLU(),
        )
        self.fc_input_size = self._get_conv_output(input_size)
        self.fc_layer = nn.Sequential(
            nn.Linear(self.fc_input_size, 100),
            nn.ReLU(),
            nn.Linear(100, output_size),
        )

    def _get_conv_output(self, input_size):
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, input_size)
            dummy_output = self.conv_layer(dummy_input)
            return dummy_output.view(1, -1).size(1)

    def forward(self, x):
        conv_out = self.conv_layer(x)
        conv_out = conv_out.view(conv_out.size(0), -1)  # 展平
        output = self.fc_layer(conv_out)
        return output

# Data Set
data_dir = './LIIUDATA' # ./ChestX-rayDATA | ./BrainTumorMRIDATA
transform1 = transforms.Compose([
    transforms.Resize((200, 200)),
    transforms.ToTensor(),
    transforms.Normalize([0], [1])
])

image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), transform=transform1)
                  for x in ['train', 'test']}

dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=100, shuffle=True)
               for x in ['train', 'test']}

dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'test']}
class_names = image_datasets['test'].classes

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# ResNet Model
class ResNet(nn.Module):
    def __init__(self, num_classes):
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(64, 64, 2)
        self.layer2 = self._make_layer(64, 128, 2)
        self.layer3 = self._make_layer(128, 256, 2, stride=5)
        self.layer4 = self._make_layer(256, 64, 2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, num_classes)

    def _make_layer(self, in_channels, out_channels, blocks, stride=1):
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride))
        for _ in range(1, blocks):
            layers.append(ResidualBlock(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        key = x
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        out = self.fc(x)
        return out, key

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels))

    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x += self.shortcut(residual)
        x = self.relu(x)
        return x

model = ResNet(num_classes=10).to(device) # num_classes=3 | num_classes=4
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

for epoch in range(15):
    model.train()
    for inputs, labels in dataloaders['train']:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs, _ = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
torch.save(model.state_dict(), "the best")

setup_seed(0)

cnn = CNNModel(3200, 10, 50).to(device)
criterion = nn.MSELoss()
optimizer1 = optim.Adam(cnn.parameters(), lr=0.001)

test_acc_history = []

for epoch in range(15):
    losses = []
    cnn.train()
    for inputs, labels in dataloaders['train']:
        labels_one_hot = one_hot_encode(labels, 10)
        inputs, labels_one_hot = inputs.to(device), labels_one_hot.to(device)
        _, key = model.forward(inputs)
        v = key.view(key.size(0), -1)
        pooled_v = torch.max_pool1d(v.unsqueeze(1), kernel_size=2).squeeze(1)
        pooled_v = pooled_v.to(device)
        output = cnn(pooled_v.unsqueeze(1))
        loss = criterion(output, labels_one_hot)
        optimizer1.zero_grad()
        loss.backward()
        optimizer1.step()
        losses.append(loss.item())
    print(np.mean(losses))
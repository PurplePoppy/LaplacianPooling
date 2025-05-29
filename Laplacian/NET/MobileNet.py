import torch
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import os
import torch.nn as nn
import torch.optim as optim

setup_seed(4)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = models.mobilenet_v3_large(weights=None)
model.classifier[3] = torch.nn.Linear(1280, 10)
model.aux_logits = False
model = model.to(device)

# Data Set
data_dir = './LIIUDATA' # ./ChestX-rayDATA | ./BrainTumorMRIDATA

transform1 = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0], [1])
])

image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), transform=transform1)
                  for x in ['train', 'test']}

dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=64, shuffle=True)
               for x in ['train', 'test']}

Max = 0.0
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for i in range(50):
    model.train()
    p = 0
    sum_loss = torch.zeros((100))
    for inputs, labels in dataloaders['train']:
        labels = one_hot_encode(labels, 10)
        inputs, labels = inputs.to(device), labels.to(device)
        output = model(inputs)
        loss = criterion(output, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        sum_loss[p] = loss
        p = p + 1
    print(torch.mean(sum_loss).item())

    if i >= 10:
        Max = model_eval(dataloaders, model, 'MobileNet.pth', Max)
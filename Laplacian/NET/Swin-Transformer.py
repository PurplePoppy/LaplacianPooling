import timm
import torch.optim as optim
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import os

setup_seed(4)

Max = 0.0

# Data Set
data_dir = './LIIUDATA' # ./ChestX-rayDATA | ./BrainTumorMRIDATA

transform1 = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0], [1])
])

image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), transform=transform1)
                  for x in ['train', 'test']}

dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=8, shuffle=True)
               for x in ['train', 'test']}

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = timm.create_model('swin_base_patch4_window7_224', num_classes=10).to(device) # num_classes=3 | num_classes=4

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

for i in range(16):
    model.train()
    p = 0
    sum_loss = torch.zeros((100))
    for inputs, labels in dataloaders['train']:
        labels = one_hot_encode(labels, 10)
        inputs, labels = inputs.to(device), labels.to(device)
        output = model.forward(inputs)
        loss = criterion(output, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        sum_loss[p] = loss
        p = p + 1
    print(torch.mean(sum_loss).item())

    if i >= 10:
        Max = model_eval(dataloaders, model, 'Swin-Transformer.pth', Max)
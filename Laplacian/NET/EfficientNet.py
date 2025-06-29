import timm
import torch.optim as optim
import os
import torch
import torch.nn as nn
from torchvision import datasets
import torchvision.transforms as transforms

Max = 0

# Data Set
data_dir = './LIIUDATA' # ./ChestX-rayDATA | ./BrainTumorMRIDATA

transform1 = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0], [1])
])

image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), transform=transform1)
                  for x in ['train', 'test']}

dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=16, shuffle=True)
               for x in ['train', 'test']}

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    torch.use_deterministic_algorithms(True)
setup_seed(0)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

model = timm.create_model('efficientnet_b0', num_classes=10).to(device) # num_classes=3 | num_classes=4

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.02)
ev = []
for i in range(40):
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
    print(torch.mean(sum_loss))

    if i>=50:
        model.eval()
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        all_test_target = []
        all_test_output = []
        m = 0
        for inputs, labels in dataloaders['val']:
            inputs, labels = inputs.to(device), labels.to(device)
            all_test_target.append(labels)
            output = model(inputs)
            predicted_class = torch.argmax(output, dim=1).to(device)
            all_test_output.append(predicted_class)
            m = m + 1
        all_test_target = torch.cat(all_test_target)
        all_test_output = torch.cat(all_test_output)
        acu = torch.sum(all_test_output == all_test_target).item() / 580.0 # 1288.0 | 1311.0
        acu_percent = acu * 100
        print(f'Accuracy: {acu_percent:.2f}%')
        print(Max)
        if acu_percent > Max:
            torch.save(model.state_dict(), os.path.join('./Best', 'EfficientNet.pth'))
            Max = acu_percent
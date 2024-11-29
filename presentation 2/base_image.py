import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import numpy as np
import pandas as pd 
from fastai.vision.all import * 
from fastai.metrics import accuracy
from PIL import Image
from PIL import ImageFile 
import matplotlib.pyplot as plt 
import torch 
from torch import nn, optim
import torch.nn.functional as F
from tqdm import tqdm 
from IPython.display import clear_output 
import warnings 
from pathlib import Path
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import TensorDataset, DataLoader, Dataset
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import optuna
from sklearn.metrics import confusion_matrix
import seaborn as sns
from torch.cuda.amp import GradScaler, autocast
warnings.filterwarnings("ignore")
# setting device on GPU if available, else CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data_path = Path(r'C:\Users\Acer\Desktop\test\Mushrooms')
def print_history(history, title):
    plt.figure(figsize=(7, 4))
    plt.plot(history)
    plt.title(title)
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    
class MushroomDataset(Dataset):
    def __init__(self, data_path, transform=None):
        self.data_path = data_path
        self.transform = transform
        self.images = []
        self.labels = []
        
        for folder_name in os.listdir(data_path):
            folder_path = os.path.join(data_path, folder_name)
            if not os.path.isdir(folder_path):
                continue
            for img_name in os.listdir(folder_path):
                img_path = os.path.join(folder_path, img_name)
                self.images.append(img_path)
                self.labels.append(folder_name)
        
        self.label_encoder = LabelEncoder()
        self.encoded_labels = self.label_encoder.fit_transform(self.labels)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.encoded_labels[idx]
        try:
            img = Image.open(img_path).convert('RGB').resize((96, 96))
            if self.transform:
                img = self.transform(img)
        except:
            print(f"Error loading image {img_path}, returning white image.")
            img = np.ones((96, 96, 3), dtype=np.uint8) * 255  # Белое изображение (255 для каждого канала RGB)
        return img, label
    
img_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.13, 0.31, 0.31], std=[0.5, 0.5, 0.5])
])

dataset = MushroomDataset(data_path, transform=img_transforms)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
del dataset
train_dataloader = DataLoader(train_dataset, 32, shuffle=False)
test_dataloader = DataLoader(test_dataset, 32, shuffle=False)

def build_alexnet():
    def init_weights(m):
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)
        if isinstance(m, torch.nn.Conv2d):
            torch.nn.init.xavier_uniform_(m.weight)
        
    net = torch.nn.Sequential(
        # Input: 96x96x3 (RGB Image)
        torch.nn.Conv2d(3, 96, 11, stride=4, padding=2),  # out: 24x24x96
        torch.nn.ReLU(),
        torch.nn.MaxPool2d(kernel_size=3, stride=2),  # out: 11x11x96
        
        torch.nn.Conv2d(96, 256, 5, padding=2),  # out: 11x11x256
        torch.nn.ReLU(),
        torch.nn.MaxPool2d(kernel_size=3, stride=2),  # out: 5x5x256
        
        torch.nn.Conv2d(256, 384, 3, padding=1),  # out: 5x5x384
        torch.nn.ReLU(),
        
        torch.nn.Conv2d(384, 384, 3, padding=1),  # out: 5x5x384
        torch.nn.ReLU(),
        
        torch.nn.Conv2d(384, 256, 3, padding=1),  # out: 5x5x256
        torch.nn.ReLU(),
        torch.nn.MaxPool2d(kernel_size=3, stride=2),  # out: 2x2x256
        
        torch.nn.Flatten(),
        
        torch.nn.Linear(2*2*256, 4096),  # fully connected layer 1
        torch.nn.ReLU(),
        torch.nn.Dropout(0.5),
        
        torch.nn.Linear(4096, 4096),  # fully connected layer 2
        torch.nn.ReLU(),
        torch.nn.Dropout(0.5),
        
        torch.nn.Linear(4096, 10)  # fully connected layer 3 (output layer, 10 classes)
    )

    net.apply(init_weights)
    return net

net = build_alexnet()
net.to(device)
#print(net)
del train_dataloader

def train(net, train_loader, device, num_epochs, learning_rate):
    optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9)
    loss_function = torch.nn.CrossEntropyLoss()
    acc_history = []
    
    scaler = GradScaler()

    with tqdm(total=len(train_loader)*num_epochs, position=0, leave=True) as pbar:

        for epoch in range(num_epochs):
            running_loss = 0.0
            correct = 0
            total = 0 
            
            for batch_num, (inputs, labels) in enumerate(train_loader):
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                optimizer.zero_grad()

                # Mixed Precision Training
                with autocast():  
                    outputs = net(inputs)
                    loss = loss_function(outputs, labels)

                # Backpropagation                
                scaler.scale(loss).backward()

            
                scaler.step(optimizer) 
                scaler.update() 
                running_loss += loss.item()

                # Calculate batch Accuracy
                _, predicted = outputs.max(1)
                batch_total = labels.size(0)
                batch_correct = predicted.eq(labels).sum().item()
                batch_acc = batch_correct/batch_total
                
                pbar.set_description("Epoch: %d, Batch: %2d, Loss: %.2f, Acc: %.2f" % (epoch, batch_num, running_loss, batch_acc))
                pbar.update()

                total += batch_total
                correct += batch_correct

            # Print the evaluation metric and reset it for the next epoch
            acc = correct/total 
            acc_history.append(acc)

        pbar.close()

    return acc_history

if __name__ == '__main__':
    EPOCHS = 15
    BATCH_SIZE = 64
    LR = 0.0023907002927538432
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE,
        shuffle = True,
        num_workers=8,
        prefetch_factor=32
    )
    hist_net = train(net, train_dataloader, device, EPOCHS, LR)

    print_history(hist_net, "AlexNet Model Accuracy")
    
    print(hist_net)
# prerequisites
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from torchvision.utils import save_image

# Device configuration
device = torch.device('mps' if torch.mps.is_available() else 'cpu')
print(device)
BATCH_SIZE = 100

# MNIST Dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(0.13, 0.31)])

train_dataset = datasets.MNIST(root='data', train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(root='data', train=False, transform=transform, download=True)

#train_dataset = datasets.FashionMNIST(root='data', train=True, transform=transform, download=True)
#test_dataset = datasets.FashionMNIST(root='data', train=False, transform=transform, download=True)

# Data Loader (Input Pipeline)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)

def init_weights(m):
    if type(m) == torch.nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)

class Generator(nn.Module):
    def __init__(self, g_input_dim, g_output_dim):
        super(Generator, self).__init__()       
        self.fc1 = nn.Linear(g_input_dim, 512)
        self.fc2 = nn.Linear(512, 1024)
        self.fc3 = nn.Linear(1024, 1024)
        self.fc4 = nn.Linear(1024, g_output_dim)
        
        self.apply(init_weights)
    
    # forward method
    def forward(self, x): 
        # x = F.leaky_relu(self.fc1(x), 0.2)
        # x = F.leaky_relu(self.fc2(x), 0.2)
        # x = F.leaky_relu(self.fc3(x), 0.2)
        # return torch.tanh(self.fc4(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return torch.tanh(self.fc4(x))
    
class Discriminator(nn.Module):
    def __init__(self, d_input_dim):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(d_input_dim, 1024)
        self.fc2 = nn.Linear(self.fc1.out_features, self.fc1.out_features//2)
        self.fc3 = nn.Linear(self.fc2.out_features, self.fc2.out_features//2)
        self.fc4 = nn.Linear(self.fc3.out_features, 1)
        
        self.apply(init_weights)
    
    # forward method
    def forward(self, x):
        x = F.leaky_relu(self.fc1(x), 0.2)
        x = F.dropout(x, 0.3)
        x = F.leaky_relu(self.fc2(x), 0.2)
        x = F.dropout(x, 0.3)
        x = F.leaky_relu(self.fc3(x), 0.2)
        x = F.dropout(x, 0.3)
        return torch.sigmoid(self.fc4(x))
    
# Условная метка
LABEL = 1
NUM_CLASSES = 10

# Параметры
Z_DIM = 50
mnist_dim = 28 * 28
BATCH_SIZE = 100
lr = 0.0002

class Generator(nn.Module):
    def __init__(self, g_input_dim, g_output_dim, num_classes):
        super(Generator, self).__init__()
        self.label_emb = nn.Embedding(num_classes, num_classes)
        self.model = nn.Sequential(
            nn.Linear(g_input_dim + num_classes, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, g_output_dim),
            nn.Tanh()
        )

    def forward(self, z, labels):
        labels_emb = self.label_emb(labels)
        x = torch.cat([z, labels_emb], dim=1)
        return self.model(x)

class Discriminator(nn.Module):
    def __init__(self, d_input_dim, num_classes):
        super(Discriminator, self).__init__()
        self.label_emb = nn.Embedding(num_classes, num_classes)
        self.model = nn.Sequential(
            nn.Linear(d_input_dim + num_classes, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x, labels):
        labels_emb = self.label_emb(labels).to(x.device)  # Перенос на устройство `x`
        x = torch.cat([x, labels_emb], dim=1)
        return self.model(x)

# Условный генератор и дискриминатор
G = Generator(g_input_dim=Z_DIM, g_output_dim=mnist_dim, num_classes=NUM_CLASSES).to(device)
D = Discriminator(d_input_dim=mnist_dim, num_classes=NUM_CLASSES).to(device)

# Оптимизаторы
G_optimizer = optim.Adam(G.parameters(), lr=lr)
D_optimizer = optim.Adam(D.parameters(), lr=lr)

# Функция ошибки
criterion = nn.BCELoss()

# Обучение дискриминатора
def D_train(x, labels):
    D.zero_grad()

    # Обработка реальных данных
    x_real, y_real = x.view(-1, mnist_dim), torch.ones(BATCH_SIZE, 1).to(device)
    labels = labels.to(device)
    x_real, y_real = Variable(x_real.to(device)), Variable(y_real)

    D_output_real = D(x_real, labels)
    D_real_loss = criterion(D_output_real, y_real)

    # Обработка фейковых данных
    z = Variable(torch.randn(BATCH_SIZE, Z_DIM).to(device))
    fake_labels = labels
    x_fake = G(z, fake_labels)
    y_fake = Variable(torch.zeros(BATCH_SIZE, 1).to(device))

    D_output_fake = D(x_fake, fake_labels)
    D_fake_loss = criterion(D_output_fake, y_fake)

    # Обновление весов дискриминатора
    D_loss = D_real_loss + D_fake_loss
    D_loss.backward()
    D_optimizer.step()

    return D_loss.item()

# Обучение генератора
def G_train(labels):
    G.zero_grad()

    z = Variable(torch.randn(BATCH_SIZE, Z_DIM).to(device))
    fake_labels = labels.to(device)
    y = Variable(torch.ones(BATCH_SIZE, 1).to(device))

    G_output = G(z, fake_labels)
    D_output = D(G_output, fake_labels)
    G_loss = criterion(D_output, y)

    # Обновление весов генератора
    G_loss.backward()
    G_optimizer.step()

    return G_loss.item()

# Генерация изображений
def generate_test_image(epoch):
    with torch.no_grad():
        z = Variable(torch.randn(1, Z_DIM).to(device))
        label = Variable(torch.tensor([LABEL]).to(device))  # Метка класса 1
        generated = G(z, label)
        save_image(generated.view(1, 1, 28, 28), f'./pract 18.11.24/output/sample_{epoch}.png')

# Обучение GAN
n_epoch = 300
for epoch in range(1, n_epoch + 1):
    D_losses, G_losses = [], []
    for batch_idx, (x, labels) in enumerate(train_loader):
        D_losses.append(D_train(x, labels))
        G_losses.append(G_train(labels))

    print(f'[{epoch}/{n_epoch}]: loss_d: {torch.mean(torch.FloatTensor(D_losses)):.3f}, loss_g: {torch.mean(torch.FloatTensor(G_losses)):.3f}')
    generate_test_image(epoch)
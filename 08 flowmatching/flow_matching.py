import torch
import torchvision
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

def load_mnist():
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = datasets.MNIST(root='../data', train=True, download=True, transform=transform)
    train_data, test_data = train_test_split(dataset, test_size=0.2, random_state=42)
    
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=64, shuffle=False)
    return trainloader, testloader

class MLP(nn.Module):
    def __init__(self, input_dim=784, hidden_dims=[512, 256, 128], output_dim=10):
        super(MLP, self).__init__()
        layers = []
        prev_dim = input_dim
        
        # Build hidden layers
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim)
            ])
            prev_dim = hidden_dim
            
        # Output layer
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.net = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.net(x)

def train_flow_matching(model, trainloader, num_epochs=10, lr=0.001):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for images, labels in tqdm(trainloader):
            images = images.view(images.size(0), -1).to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
        epoch_loss = running_loss / len(trainloader)
        accuracy = 100. * correct / total
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {accuracy:.2f}%')

def main():
    # Load data
    trainloader, testloader = load_mnist()
    
    # Create and train the model
    model = MLP()
    train_flow_matching(model, trainloader)

if __name__ == "__main__":
    main() 
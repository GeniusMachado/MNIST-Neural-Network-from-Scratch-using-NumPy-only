import torch # this is the magic under the hood ! It starts from here where we start leveraging your GPU
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def prepare_data(path):
    df = pd.read_csv(path)
    data = df.values
    np.random.shuffle(data)
    
    train_data = data[1000:]
    dev_data = data[:1000]
    
    def scale(subset):
        y = torch.tensor(subset[:, 0], dtype=torch.long)
        x = torch.tensor(subset[:, 1:], dtype=torch.float32) / 255.0
        return x, y

    x_train, y_train = scale(train_data)
    x_dev, y_dev = scale(dev_data)
    
    return x_train, y_train, x_dev, y_dev

class DigitNet(nn.Module):
    def __init__(self):
        super(DigitNet, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(784, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        return self.network(x)

def train_model(x_train, y_train, epochs=50, batch_size=64, lr=0.001):
    model = DigitNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    dataset = TensorDataset(x_train, y_train)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch_x, batch_y in loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        if epoch % 10 == 0:
            print(f"Epoch {epoch} | Loss: {total_loss/len(loader):.4f}")
            
    return model

def validate(model, x_dev, y_dev):
    model.eval()
    with torch.no_grad():
        x_dev, y_dev = x_dev.to(device), y_dev.to(device)
        outputs = model(x_dev)
        predictions = torch.argmax(outputs, dim=1)
        accuracy = (predictions == y_dev).float().mean()
        print(f"Dev Set Accuracy: {accuracy.item() * 100:.2f}%")

if __name__ == "__main__":
    xt, yt, xv, yv = prepare_data('/kaggle/input/digit-recognizer/train.csv')
    mnist_model = train_model(xt, yt)
    validate(mnist_model, xv, yv)

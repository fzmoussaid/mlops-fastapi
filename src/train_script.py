import torch
from torch import nn
import torch.nn.functional as F
import os
import torch.optim as optim
from data_loading import get_data

os.environ['CUDA_LAUNCH_BLOCKING']="1"
os.environ['TORCH_USE_CUDA_DSA'] = "1"

class CNNModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.maxPool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16*9*9, 120)
        self.fc2 = nn.Linear(120, 60)
        self.fc3 = nn.Linear(60, 15)

    def forward(self, x):
        x = self.maxPool(F.relu(self.conv1(x)))
        x = self.maxPool(F.relu(self.conv2(x)))
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def set_training_device():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return device

def save_model(cnn, path):
    torch.save(cnn.state_dict(), path)

def save_entire_model(model, path):
    model_scripted = torch.jit.script(model)
    model_scripted.save(path)

if __name__ == "__main__":
    dataset_path = "./Vegetable Images"
    dataset_type = "/train"
    train_loader, _ = get_data(dataset_path, dataset_type)
    device = set_training_device()
    cnn = CNNModel()
    cnn.to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(cnn.parameters(), lr=0.001)
    nb_epochs = 30
    running_loss = 0.0
    for epoch in range(nb_epochs):
        for i, data in enumerate(train_loader):
            image_batch, labels = data
            image_batch, labels = data[0].to(device), data[1].to(device)
            optimizer.zero_grad()
            res = cnn.forward(image_batch).to(device)
            l = loss_fn(res, labels)
            l.backward()
            optimizer.step()
            running_loss += l.item()
            if i % 5 == 0:
                print("Epoch {}, Iter {} : loss {}".format(epoch, i, running_loss))
                running_loss = 0.0
    # Save model as a state dictionary for a quick model evaluation
    save_model(cnn, "test_classification_model.pth")
    # Save scripted model to use for inference
    save_entire_model(cnn , "vegetables_classification_net.pth")
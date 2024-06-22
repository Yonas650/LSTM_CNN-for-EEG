import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import mat73
import numpy as np
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import os

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=1)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=1)
        self.bn2 = nn.BatchNorm1d(out_channels)
        if in_channels != out_channels:
            self.adjust_residual = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        else:
            self.adjust_residual = None
        
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.adjust_residual:
            residual = self.adjust_residual(residual)
        out += residual
        out = self.relu(out)
        return out

class CNN_LSTM(nn.Module):
    def __init__(self):
        super(CNN_LSTM, self).__init__()
        self.cnn = nn.Sequential(
            ResidualBlock(59, 64),
            nn.MaxPool1d(kernel_size=2),
            ResidualBlock(64, 128),
            nn.MaxPool1d(kernel_size=2),
            ResidualBlock(128, 256),
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout(0.5)
        )
        self.lstm = nn.LSTM(256, 256, batch_first=True, num_layers=2, dropout=0.5)
        self.attention = nn.MultiheadAttention(embed_dim=256, num_heads=8)
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, 4)
        self.dropout = nn.Dropout(0.5)
        self.relu = nn.ReLU()

    def forward(self, x):
        batch_size = x.size(0)
        x = self.cnn(x)
        x = x.permute(2, 0, 1)  #(time, batch, channels)
        x, _ = self.attention(x, x, x)
        x = x.permute(1, 0, 2)  #(batch, time, channels)
        h_0 = torch.zeros(2, batch_size, 256).to(x.device)
        c_0 = torch.zeros(2, batch_size, 256).to(x.device)
        _, (h_n, _) = self.lstm(x, (h_0, c_0))
        x = self.dropout(self.relu(self.fc1(h_n[-1])))
        x = self.fc2(x)
        return x

def save_heatmap(cm, iteration, accuracy, output_dir='output_swap'):
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='.2f', cmap='Blues')
    plt.title(f'Confusion Matrix - Iteration {iteration} - Accuracy: {accuracy:.2f}%')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, f'confusion_matrix_iter_{iteration}.png'))
    plt.close()


thermal_dict = mat73.loadmat('data_PS_SR.mat')
thermal_tr_con = thermal_dict.get('data_m', None)

y_thermal_dict = mat73.loadmat('y_PS.mat')
y_thermal = y_thermal_dict.get('y_PS', None)

if thermal_tr_con is None or y_thermal is None:
    raise ValueError("Failed to load the data. Please check the data files.")

print(f"Original shape of thermal_tr_con: {np.shape(thermal_tr_con)}")
print(f"Original shape of y_thermal: {np.shape(y_thermal)}")

X = np.array(thermal_tr_con)
y = np.array(y_thermal)

X = (X - np.mean(X, axis=3, keepdims=True)) / np.std(X, axis=3, keepdims=True)
X = X[:, :, :, 600:1600, :]

print(f"Shape of X after cropping: {X.shape}")
if X.shape[3] != 1000:
    raise ValueError(f"Unexpected shape after cropping: {X.shape}")

#swap axes to bring trials to the front
X = X.swapaxes(1, 4).squeeze(-1)  # (28, 80, 59, 1000)

#flatten the labels to match the new X shape using swapaxes
y = y.swapaxes(1, 2).squeeze(-1)  #(28, 80)

#verify the shapes
print(f"Shape of X after swapping axes: {X.shape}")
print(f"Shape of y after swapping axes: {y.shape}")

#the data is in the shape (28, 80, 59, 1000) and (28, 80) respectively, 
# with trials kept separate.

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

loo = LeaveOneOut()
all_test_accuracies = []
all_y_true = []
all_y_pred = []
iteration = 0

for train_index, test_index in loo.split(np.arange(X.shape[0])):
    iteration += 1
    
    #select train and test subjects
    X_train = X[train_index].reshape(-1, 59, 1000)
    y_train = y[train_index].reshape(-1)
    X_test = X[test_index].reshape(-1, 59, 1000)
    y_test = y[test_index].reshape(-1)

    if len(y_test) == 0:
        print(f"Skipping iteration {iteration} due to empty test set.")
        continue

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long).to(device)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long).to(device)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)  
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    model = CNN_LSTM().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.0001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.001, steps_per_epoch=len(train_loader), epochs=200)

    num_epochs = 200 
    best_val_accuracy = 0
    patience = 15 
    trigger_times = 0

    for epoch in range(num_epochs):
        model.train()
        train_correct = 0
        train_total = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

        train_accuracy = 100 * train_correct / train_total

        model.eval()
        test_correct = 0
        test_total = 0
        y_pred_iter = []
        y_true_iter = []
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                test_total += labels.size(0)
                test_correct += (predicted == labels).sum().item()
                y_pred_iter.extend(predicted.cpu().numpy())
                y_true_iter.extend(labels.cpu().numpy())

        test_accuracy = 100 * test_correct / test_total if test_total > 0 else 0
        print(f'Epoch [{epoch + 1}/{num_epochs}], Train Accuracy: {train_accuracy:.2f}%, Test Accuracy: {test_accuracy:.2f}%')

        if test_accuracy > best_val_accuracy:
            best_val_accuracy = test_accuracy
            trigger_times = 0
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            trigger_times += 1
            if trigger_times >= patience:
                print('Early stopping!')
                break

    model.load_state_dict(torch.load('best_model.pth'))

    model.eval()
    test_correct = 0
    test_total = 0
    y_pred = []
    y_true = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            test_total += labels.size(0)
            test_correct += (predicted == labels).sum().item()
            y_pred.extend(predicted.cpu().numpy())
            y_true.extend(labels.cpu().numpy())

    test_accuracy = 100 * test_correct / test_total if test_total > 0 else 0
    all_test_accuracies.append(test_accuracy)
    all_y_true.extend(y_true)
    all_y_pred.extend(y_pred)
    print(f'Test Accuracy: {test_accuracy:.2f}%')

    cm = confusion_matrix(y_true, y_pred)
    cm = (cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]) * 100
    save_heatmap(cm, iteration, test_accuracy)

average_accuracy = np.mean(all_test_accuracies) if all_test_accuracies else 0
print(f'Average Test Accuracy: {average_accuracy:.2f}%')

if all_test_accuracies:
    cm = confusion_matrix(all_y_true, all_y_pred)
    cm = (cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]) * 100
    save_heatmap(cm, 'average', average_accuracy)
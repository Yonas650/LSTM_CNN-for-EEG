import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
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

def train_test_loso(lo_sub, path="np_files", n_sub=28):
    #load the test data
    X_test_path = os.path.join(path, f'X_PS_SR_{lo_sub}.npy')
    y_test_path = os.path.join(path, f'y_PS_{lo_sub}.npy')
    X_test = np.load(X_test_path).squeeze(-1)  #remove the extra dimension
    y_test = np.load(y_test_path)
    
    #load the training data
    X_train = []
    y_train = []
    for sub_id in range(n_sub):
        if sub_id != lo_sub:
            X_path = os.path.join(path, f'X_PS_SR_{sub_id}.npy')
            y_path = os.path.join(path, f'y_PS_{sub_id}.npy')
            X_train.append(np.load(X_path).squeeze(-1))  #remove the extra dimension
            y_train.append(np.load(y_path))
    
    #concatenate all training data
    X_train = np.concatenate(X_train, axis=0)
    y_train = np.concatenate(y_train, axis=0)
    
    return X_train, y_train, X_test, y_test


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

all_test_accuracies = []
all_y_true = []
all_y_pred = []

for subject_id in range(28):
    print(f"Training with subject {subject_id} left out as test set")
    
    X_train, y_train, X_test, y_test = train_test_loso(subject_id)

    #convert data to PyTorch tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long).to(device)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long).to(device)

    
    print(f"X_train_tensor shape: {X_train_tensor.shape}")

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
    print(f'Test Accuracy for subject {subject_id}: {test_accuracy:.2f}%')

    cm = confusion_matrix(y_true, y_pred)
    cm = (cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]) * 100
    save_heatmap(cm, subject_id, test_accuracy)

average_accuracy = np.mean(all_test_accuracies) if all_test_accuracies else 0
print(f'Average Test Accuracy: {average_accuracy:.2f}%')

if all_test_accuracies:
    cm = confusion_matrix(all_y_true, all_y_pred)
    cm = (cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]) * 100
    save_heatmap(cm, 'average', average_accuracy)

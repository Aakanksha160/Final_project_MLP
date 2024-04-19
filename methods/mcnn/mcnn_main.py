import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, f1_score, average_precision_score

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=10, out_channels=32, kernel_size=3)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3)
        self.fc1 = nn.Linear(64 * 8, 128)
        self.fc2 = nn.Linear(128, 2) 

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool1d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool1d(x, 2)
        x = x.view(-1, 64 * 8)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def mcnn_main(
    train_feature,
    train_label,
    test_feature,
    test_label,
    epochs=30,
    batch_size=512,
    lr=0.01,  # Learning rate for SGD optimizer
    momentum=0.9,  # Momentum for SGD optimizer
    device="cpu"
):
    train_feature = torch.from_numpy(train_feature).to(dtype=torch.float32).to(device)
    train_label = torch.from_numpy(train_label).to(dtype=torch.long).to(device)
    test_feature = torch.from_numpy(test_feature).to(dtype=torch.float32).to(device)
    test_label = torch.from_numpy(test_label).to(dtype=torch.long).to(device)

    model = SimpleCNN()
    model.to(device)

    # Define loss function
    loss_func = nn.CrossEntropyLoss()

    # Define optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)

    for epoch in range(epochs):
        # Training
        model.train()
        total_loss = 0.
        for i in range(0, len(train_label), batch_size):
            optimizer.zero_grad()
            batch_feature = train_feature[i:i+batch_size]
            batch_label = train_label[i:i+batch_size]

            output = model(batch_feature)
            loss = loss_func(output, batch_label)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        # Compute training metrics
        train_pred = torch.argmax(model(train_feature), dim=1).cpu().numpy()
        train_true = train_label.cpu().numpy()
        train_auc = roc_auc_score(train_true, train_pred)
        train_f1 = f1_score(train_true, train_pred, average='macro')
        train_ap = average_precision_score(train_true, train_pred)

        # Testing
        model.eval()
        test_pred = []
        with torch.no_grad():
            for i in range(0, len(test_label), batch_size):
                batch_feature = test_feature[i:i+batch_size]
                output = model(batch_feature)
                pred = torch.argmax(output, dim=1).cpu().numpy()
                test_pred.extend(pred)

        # Compute testing metrics
        test_true = test_label.cpu().numpy()
        test_auc = roc_auc_score(test_true, test_pred)
        test_f1 = f1_score(test_true, test_pred, average='macro')
        test_ap = average_precision_score(test_true, test_pred)

        print(f"Epoch {epoch+1}/{epochs}: "
              f"Train Loss: {total_loss / len(train_label):.4f}, "
              f"Train AUC: {train_auc:.4f}, Train F1: {train_f1:.4f}, Train AP: {train_ap:.4f}, "
              f"Test AUC: {test_auc:.4f}, Test F1: {test_f1:.4f}, Test AP: {test_ap:.4f}")

    return model

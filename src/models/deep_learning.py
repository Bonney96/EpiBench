# src/models/deep_learning.py

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class SeqCNNRegressor(nn.Module):
    def __init__(self, 
                 kernel_sizes=(3,5,7),
                 filters_per_branch=64, 
                 fc_units=64, 
                 dropout_rate=0.5,
                 use_batchnorm=True):
        super(SeqCNNRegressor, self).__init__()
        
        self.kernel_sizes = kernel_sizes
        self.use_batchnorm = use_batchnorm

        # We'll dynamically create the convolution branches based on kernel_sizes
        self.branches = nn.ModuleList()
        self.branch_norms = nn.ModuleList()
        in_channels = 4  # For one-hot encoded sequence 'A,C,G,T'
        for k in self.kernel_sizes:
            conv = nn.Conv1d(in_channels=in_channels, out_channels=filters_per_branch, kernel_size=k, padding=k//2)
            self.branches.append(conv)
            if self.use_batchnorm:
                self.branch_norms.append(nn.BatchNorm1d(filters_per_branch))
            else:
                self.branch_norms.append(nn.Identity())
        
        # After concatenation: total_channels = filters_per_branch * len(kernel_sizes)
        total_channels = filters_per_branch * len(self.kernel_sizes)
        
        # Additional convolution block
        self.conv2 = nn.Conv1d(in_channels=total_channels, out_channels=total_channels*2, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(total_channels*2) if self.use_batchnorm else nn.Identity()
        self.pool = nn.MaxPool1d(2)
        
        self.conv3 = nn.Conv1d(in_channels=total_channels*2, out_channels=total_channels*4, kernel_size=7, padding=3)
        self.bn3 = nn.BatchNorm1d(total_channels*4) if self.use_batchnorm else nn.Identity()
        
        self.fc1 = nn.Linear(total_channels*4, fc_units)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(fc_units, 1)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # x shape: (batch, 4, seq_len)
        branch_outputs = []
        for conv, bn in zip(self.branches, self.branch_norms):
            out = conv(x)
            out = bn(out)
            out = self.relu(out)
            branch_outputs.append(out)

        # Concatenate along channels
        x_cat = torch.cat(branch_outputs, dim=1)
        
        x = self.conv2(x_cat)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.pool(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)

        # Global average pooling
        x = x.mean(dim=2)  # (batch, channels)
        
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


def train_model(model, train_loader, val_loader, epochs=50, lr=0.001, weight_decay=1e-5, device=None, early_stop_patience=5):
    """
    Train the given model using the provided train and validation data loaders.

    Parameters:
        model (nn.Module): The regression model to train.
        train_loader (DataLoader): Loader providing (X, y, mask) for training.
        val_loader (DataLoader): Loader providing (X, y, mask) for validation.
        epochs (int): Maximum number of training epochs.
        lr (float): Learning rate for the optimizer.
        weight_decay (float): Weight decay (L2 regularization) for the optimizer.
        device (torch.device): The device to use for training.
        early_stop_patience (int): Early stopping patience in epochs.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(epochs):
        # Training phase
        model.train()
        total_train_loss = 0.0
        for X_batch, y_batch, mask_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device).unsqueeze(1)  # Make target shape (batch, 1)

            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)

        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch, mask_batch in val_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device).unsqueeze(1)
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)

        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

        # Early stopping check
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), "best_model.pth")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= early_stop_patience:
                print("Early stopping triggered.")
                break


def predict_model(model, test_loader, device=None):
    """
    Generate predictions for the data provided by the test_loader.

    Parameters:
        model (nn.Module): The trained regression model.
        test_loader (DataLoader): Loader providing (X, y, mask) for the test set.
        device (torch.device): The device to use for inference.

    Returns:
        np.ndarray: The model predictions as a numpy array of shape (num_samples,).
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    preds = []
    with torch.no_grad():
        for X_batch, y_batch, mask_batch in test_loader:
            X_batch = X_batch.to(device)
            outputs = model(X_batch)  # (batch, 1)
            preds.extend(outputs.squeeze(1).cpu().numpy())
    return np.array(preds)

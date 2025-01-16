# deep_learning.py

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import math

PAD_IDX = 4  # Same as in datasets.py

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=10000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # shape (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len, :]
        return self.dropout(x)

class DNARegressor(nn.Module):
    def __init__(self, vocab_size=5, embed_dim=64, num_heads=4, num_layers=4, ff_dim=256, dropout=0.1):
        super(DNARegressor, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=PAD_IDX)
        self.pos_encoder = PositionalEncoding(d_model=embed_dim, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim,
                                                   nhead=num_heads,
                                                   dim_feedforward=ff_dim,
                                                   dropout=dropout,
                                                   batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.fc1 = nn.Linear(embed_dim, 64)
        self.fc2 = nn.Linear(64, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, attention_mask):
        # x: (batch, seq_len)
        # attention_mask: (batch, seq_len), True for token, False for pad
        # For Transformer, src_key_padding_mask: True means this token is pad.
        src_key_padding_mask = ~attention_mask  # invert mask
        emb = self.embedding(x)  # (batch, seq_len, embed_dim)
        emb = self.pos_encoder(emb)
        out = self.transformer_encoder(emb, src_key_padding_mask=src_key_padding_mask)  # (batch, seq_len, embed_dim)

        # Mean pool over non-padding tokens
        mask = attention_mask.unsqueeze(-1)
        out = out * mask
        sum_out = out.sum(dim=1)
        lengths = mask.sum(dim=1)
        pooled = sum_out / lengths.clamp(min=1)

        # Regression head
        x = self.relu(self.fc1(pooled))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

def train_model(model, train_loader, val_loader, epochs=20, lr=0.001, weight_decay=1e-5, device=None, early_stop_patience=5):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif isinstance(device, str):
        device = torch.device(device)

    model.to(device)

    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scaler = torch.amp.GradScaler(enabled=(device.type == 'cuda'))

    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        total_train_loss = 0.0
        for X_batch, y_batch, mask_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device).unsqueeze(1)
            mask_batch = mask_batch.to(device)

            optimizer.zero_grad()
            with torch.amp.autocast(device_type='cuda', enabled=(device.type=='cuda')):
                outputs = model(X_batch, mask_batch)
                loss = criterion(outputs, y_batch)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch, mask_batch in val_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device).unsqueeze(1)
                mask_batch = mask_batch.to(device)
                with torch.amp.autocast(device_type='cuda', enabled=(device.type=='cuda')):
                    outputs = model(X_batch, mask_batch)
                    loss = criterion(outputs, y_batch)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), "best_model.pth")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= early_stop_patience:
                print("Early stopping triggered.")
                break

    model.load_state_dict(torch.load("best_model.pth"))

def predict_model(model, test_loader, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif isinstance(device, str):
        device = torch.device(device)

    model.to(device)
    model.eval()
    preds = []
    with torch.no_grad():
        for X_batch, _, mask_batch in test_loader:
            X_batch = X_batch.to(device)
            mask_batch = mask_batch.to(device)
            with torch.amp.autocast(device_type='cuda', enabled=(device.type=='cuda')):
                outputs = model(X_batch, mask_batch)
            preds.extend(outputs.squeeze(1).cpu().numpy())
    return np.array(preds)

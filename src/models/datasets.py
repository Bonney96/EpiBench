import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np

class SequenceDataset(Dataset):
    """
    A PyTorch Dataset for DNA sequences for regression tasks without fixed length,
    but with a maximum allowed length to prevent memory issues.
    """
    def __init__(self, csv_path):
        self.df = pd.read_csv(csv_path)
        self.mapping = {'A':0, 'C':1, 'G':2, 'T':3, 'a':0, 'c':1, 'g':2, 't':3}
        self.max_allowed_length = 10000  # Maximum allowed sequence length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        seq = row['sequence'].upper()

        # Truncate the sequence if it exceeds the max allowed length
        if len(seq) > self.max_allowed_length:
            seq = seq[:self.max_allowed_length]

        seq_len = len(seq)
        one_hot = np.zeros((4, seq_len), dtype=np.float32)
        for i, base in enumerate(seq):
            base_idx = self.mapping.get(base, 0)
            one_hot[base_idx, i] = 1.0

        y = float(row['score'])
        return torch.tensor(one_hot), torch.tensor(y, dtype=torch.float32)


def collate_fn(batch):
    seqs = [item[0] for item in batch]  # list of Tensors (4, seq_len)
    targets = [item[1] for item in batch]  # list of Tensors (score)

    max_len = max(seq.shape[1] for seq in seqs)

    padded_seqs = []
    masks = []
    for seq in seqs:
        seq_len = seq.shape[1]
        # Create a mask of length max_len: 1 for real positions, 0 for padding
        mask = torch.zeros((max_len,), dtype=torch.float32)
        mask[:seq_len] = 1.0

        if seq_len < max_len:
            pad_amount = max_len - seq_len
            pad_tensor = torch.zeros((4, pad_amount), dtype=seq.dtype)
            seq = torch.cat([seq, pad_tensor], dim=1)

        padded_seqs.append(seq)
        masks.append(mask)

    X = torch.stack(padded_seqs, dim=0)   # (batch, 4, max_len)
    y = torch.stack(targets, dim=0)       # (batch,)
    mask = torch.stack(masks, dim=0)      # (batch, max_len)

    return X, y, mask

# feature_engineering.py
import pandas as pd
from src.config import PREPARED_DATA_CSV, FEATURE_DATA_CSV
from src.utils.seq_utils import gc_content, cpg_density, kmer_counts
from itertools import product

df = pd.read_csv(PREPARED_DATA_CSV)

all_features = []
for _, row in df.iterrows():
    seq = row['sequence']
    features = {
        'chrom': row['chrom'],
        'start': row['start'],
        'end': row['end'],
        'score': row['score'],
        'seq_length': len(seq),
        'gc_content': gc_content(seq),
        'cpg_density': cpg_density(seq)
    }
    
    # Add 2-mer frequencies
    k2_counts = kmer_counts(seq, k=2)
    for kmer, val in k2_counts.items():
        features[f"k2_{kmer}"] = val
    
    # Add 3-mer frequencies
    k3_counts = kmer_counts(seq, k=3)
    for kmer, val in k3_counts.items():
        features[f"k3_{kmer}"] = val
    
    all_features.append(features)

feature_df = pd.DataFrame(all_features)
feature_df.to_csv(FEATURE_DATA_CSV, index=False)

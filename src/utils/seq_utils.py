# seq_utils.py
from itertools import product

def gc_content(seq):
    seq = seq.upper()
    gc_count = seq.count('G') + seq.count('C')
    return gc_count / len(seq) if len(seq) > 0 else 0

def cpg_density(seq):
    seq = seq.upper()
    cpg_count = seq.count("CG")
    return cpg_count / len(seq) if len(seq) > 0 else 0

def kmer_counts(seq, k=2):
    seq = seq.upper()
    counts = {}
    possible_kmers = [''.join(p) for p in product('ACGT', repeat=k)]
    for km in possible_kmers:
        counts[km] = 0
    
    for i in range(len(seq)-k+1):
        kmer = seq[i:i+k]
        if kmer in counts:
            counts[kmer] += 1
    
    length = len(seq)
    if length > 0:
        for km in counts:
            counts[km] = counts[km] / (length - k + 1)
    return counts

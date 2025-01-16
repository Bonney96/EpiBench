import sys
from Bio import SeqIO

# Update these paths as needed
bed_file = "/storage1/fs1/dspencer/Active/spencerlab/abonney/regions/simple_project/ND150826-CD34-wgbs.positive_control_regions.meth.bed"
fasta_file = "/storage1/fs1/dspencer/Active/spencerlab/abonney/regions/simple_project/positive_control_sequences.fa"
output_csv = "positive_control_data.csv"

# Read BED data
# Based on the data you showed, columns are:
# 0: chrom
# 1: start
# 2: end
# 3: coverage (not needed)
# 4: score (between 0 and 1)
regions = []
with open(bed_file, 'r') as bf:
    for line in bf:
        if line.strip():
            parts = line.strip().split('\t')
            chrom = parts[0]
            start = parts[1]
            end = parts[2]
            # coverage = parts[3]  # If you need it, but not required now
            score = float(parts[4])
            regions.append((chrom, start, end, score))

# Create a dictionary keyed by region string (chr:start-end) to score
score_dict = {}
for (chrom, start, end, score) in regions:
    region_key = f"{chrom}:{start}-{end}"
    score_dict[region_key] = score

# Parse FASTA and match sequences to scores
with open(output_csv, 'w') as out:
    out.write("chrom,start,end,score,sequence\n")
    for record in SeqIO.parse(fasta_file, "fasta"):
        # record.id should be something like: chr1:10000-10100
        region = record.id
        seq = str(record.seq)
        chrom = region.split(':')[0]
        coords = region.split(':')[1]
        start = coords.split('-')[0]
        end = coords.split('-')[1]
        score = score_dict.get(region, "unknown")
        out.write(f"{chrom},{start},{end},{score},{seq}\n")

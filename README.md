# EpiBench
Python-based toolkit designed to streamline epigenomic data analysis, particularly focusing on DNA methylation prediction.

**Data Preparation**: Automated scripts for reading BED and FASTA files and generating a consolidated CSV of sequence and methylation data.

**Feature Engineering**: Utilities to compute GC content, CpG density, and k-mer frequencies for downstream modeling.

**Modeling Pipelines**: End-to-end workflows for both regression (predicting continuous methylation values) and classification (predicting whether methylation levels exceed a threshold).

**Deep Learning**: Convolutional neural network (CNN) and Transformer-based models for sequence-based methylation prediction.

**Explainability**: Integrated Gradients (IG) to interpret model predictions at the nucleotide level, and motif discovery to identify enriched sequence patterns influencing predictions.

**Comprehensive Reporting**: Automatic generation of HTML reports summarizing data exploration, model performance, explainability results, and motif discovery findings.

[Multi-Branch_CNN_Regressor.pdf](https://github.com/user-attachments/files/18511413/Multi-Branch_CNN_Regressor.pdf)

We used GenPept-Curated-2025, obtained from the
[official GitHub repository at commit f1371433](https://github.com/biochem-data-sci/GenPept-Curated-2025/tree/f137143342eef36c5c4bde5fed16219fd2879fd0).
The dataset contains 11,000 peptide sequences, comprising 5,500 AMP
and 5,500 non-AMP samples.

Data partitioning followed the workflow provided at this fixed commit.
CD-HIT was used for clustering with a sequence identity threshold of 0.90
and a minimum coverage threshold of 0.80 for the shorter sequence.
Using a random seed of 42, clusters were assigned intact to the training,
validation, and test sets, with stratification by class label and sequence
length bin, yielding 7,700, 990, and 2,310 sequences, respectively.

All comparison methods used the same data partition.
Artificial label noise was introduced only into the training set.

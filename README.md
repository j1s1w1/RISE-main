# RISE

**Reliability-Aware Learning from Imperfect Sequence-Structure Evidence for Antimicrobial Peptide Prediction**

RISE is a sequence-structure learning framework for robust antimicrobial peptide (AMP) prediction under noisy labels and unreliable predicted structures.

It combines:

- ESM2 residue-level sequence representations;
- ESMFold-predicted residue graphs and pLDDT confidence;
- sequence and structure evidential learning;
- neighborhood-based noisy-label candidate identification;
- reliability-adjusted structural evidence;
- verified multi-source soft pseudo-label refinement.

## Repository Structure

```text
RISE/
├── train.py
├── test.py
├── configs/
│   ├── xuamp.yaml
│   ├── genpept.yaml
│   └── paths.example.yaml
├── data/
│   ├── XUAMP/
│   ├── GenPept-Curated-2025/
│   ├── fasta.py
│   └── csv_dataset.py
├── features/
│   ├── sequence_features.py
│   └── structure_features.py
├── data_preparation/
│   └── generate_symmetric_noise.py
├── requirements.txt
├── LICENSE
└── README.md
```

## Data

The repository contains the generated CSV splits used in the controlled label-noise experiments:

```text
data/<dataset>/noise_<rate>/rep<id>/
├── train.csv
├── val.csv
├── test.csv
├── noise_indices.csv
└── metadata.json
```

The training CSV contains:

| Column | Description |
|---|---|
| `sequence` | Peptide sequence |
| `clean_target` | Original clean label |
| `target` | Label used for training |
| `is_noisy` | Whether the label was flipped |

Training must use `target`. Validation and test labels remain clean.

Noise rates are `0.0`, `0.1`, `0.2`, and `0.3`, with five repetitions using seeds `42–46`.

## External Resources

The following large files are not included:

- ESM2-650M weights;
- ESMFold PDB structures;
- extracted feature caches;
- model checkpoints and logs.

Copy the path template:

```bash
cp configs/paths.example.yaml configs/paths.yaml
```

Then set the local ESM2 and PDB paths in `configs/paths.yaml`.

## Installation

```bash
conda create -n rise python=3.10 -y
conda activate rise
pip install -r requirements.txt
```

## Training

### XUAMP

```bash
CUDA_VISIBLE_DEVICES=0 python train.py \
  --config configs/xuamp.yaml \
  --paths configs/paths.yaml \
  --noise 0.100 \
  --rep 1
```

### GenPept-Curated-2025

```bash
CUDA_VISIBLE_DEVICES=0 python train.py \
  --config configs/genpept.yaml \
  --paths configs/paths.yaml \
  --noise 0.100 \
  --rep 1
```

Supported noise rates:

```text
0.000, 0.100, 0.200, 0.300
```

Supported repetitions:

```text
1, 2, 3, 4, 5
```

## Testing

```bash
CUDA_VISIBLE_DEVICES=0 python test.py \
  --config configs/xuamp.yaml \
  --paths configs/paths.yaml \
  --noise 0.100 \
  --rep 1
```

Replace `xuamp.yaml` with `genpept.yaml` for GenPept-Curated-2025.

## Regenerating Label Noise

```bash
python data_preparation/generate_symmetric_noise.py --help
```

The script:

- modifies training labels only;
- keeps validation and test labels clean;
- flips the same number of AMP and non-AMP labels;
- generates nested 10%, 20%, and 30% noisy subsets within each repetition.

## Main Results

Results are reported as mean ± standard deviation over five runs.

| Dataset | Clean ACC | Clean AUC | Clean MCC | MCC@10% | MCC@20% | MCC@30% |
|---|---:|---:|---:|---:|---:|---:|
| XUAMP | 0.750 ± 0.009 | 0.803 ± 0.005 | 0.531 ± 0.012 | 0.503 ± 0.007 | 0.485 ± 0.011 | 0.466 ± 0.024 |
| GenPept-Curated-2025 | 0.930 ± 0.002 | 0.974 ± 0.002 | 0.863 ± 0.004 | 0.847 ± 0.005 | 0.829 ± 0.005 | 0.753 ± 0.006 |

## Citation

```bibtex
@article{liu2026rise,
  title   = {Rethinking Sequence--Structure Learning from Unreliable Evidence for Antimicrobial Peptide Prediction},
  author  = {Liu, Fei and Jiang, Shouwei and Wu, Le and Lu, Wenjie and Wang, Feilong and Wu, Guangzhou and Wu, Han and Ji, Shengwei and Hong, Richang},
  year    = {2026},
  note    = {Manuscript}
}
```

## License

See [LICENSE](LICENSE).

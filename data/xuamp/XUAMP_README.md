# XUAMP Dataset Preparation

The XUAMP dataset is used in the RISE experiments, but the dataset files are **not redistributed in this repository**.

To reproduce the XUAMP experiments reported in the manuscript, please first obtain the original XUAMP dataset from its official source, then prepare the predefined train/validation/test split and generate the controlled label-noise versions used by RISE.

## Expected Directory Structure

After preparation, place the generated files under:

```text
data/XUAMP/
├── noise_0.000/
│   ├── rep1/
│   │   ├── train.csv
│   │   ├── val.csv
│   │   ├── test.csv
│   │   ├── noise_indices.csv
│   │   └── metadata.json
│   ├── rep2/
│   ├── rep3/
│   ├── rep4/
│   └── rep5/
├── noise_0.100/
│   ├── rep1/
│   ├── rep2/
│   ├── rep3/
│   ├── rep4/
│   └── rep5/
├── noise_0.200/
│   ├── rep1/
│   ├── rep2/
│   ├── rep3/
│   ├── rep4/
│   └── rep5/
└── noise_0.300/
    ├── rep1/
    ├── rep2/
    ├── rep3/
    ├── rep4/
    └── rep5/
```

## Label Convention

The predefined CSV files follow the RISE data protocol:

- `target`: observed label used for training;
- `clean_target`: original clean label;
- `is_noisy`: whether `target` differs from `clean_target`;
- `split`: train / validation / test split when included.

Controlled symmetric label noise is applied **only to the training split**.

Validation and test labels remain clean.

## Noise Settings

RISE evaluates the following training-label noise rates:

```text
0.0
0.1
0.2
0.3
```

Each noise setting is repeated five times.

The repository provides the noise-generation script:

```bash
python data_preparation/generate_symmetric_noise.py --help
```

Use this script after preparing the clean XUAMP train/validation/test split.

## Structural Data

RISE also requires precomputed ESMFold structures for XUAMP.

The PDB files are not included in this repository. Configure their local path in:

```text
configs/paths.yaml
```

For example:

```yaml
datasets:
  xuamp:
    root: ./data/XUAMP
    pdb_dir: /path/to/xuamp/esmfold_pdb
```

## Training Example

After the XUAMP files and ESMFold structures are prepared, one experiment can be launched with:

```bash
CUDA_VISIBLE_DEVICES=0 python train.py \
  --config configs/xuamp.yaml \
  --paths configs/paths.yaml \
  --noise 0.100 \
  --rep 1
```

## Testing Example

```bash
CUDA_VISIBLE_DEVICES=0 python test.py \
  --config configs/xuamp.yaml \
  --paths configs/paths.yaml \
  --noise 0.100 \
  --rep 1
```

## Note

This repository provides the RISE implementation, experiment configuration, and controlled-noise generation pipeline required to reproduce the XUAMP experiments, while the original XUAMP data should be obtained separately from its official source.

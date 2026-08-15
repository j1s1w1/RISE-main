# RISE

**Reliability-Aware Learning from Imperfect Sequence-Structure Evidence**

Official implementation for:

> **Rethinking Sequence--Structure Learning from Unreliable Evidence for Antimicrobial Peptide Prediction**

RISE is a reliability-aware sequence--structure learning framework for antimicrobial peptide (AMP) prediction when both supervision and predicted structural evidence may be unreliable.

The method uses peptide sequence as the only raw input. Frozen ESM2 residue representations and ESMFold-predicted residue graphs are encoded into sequence and structural evidence, which are fused for AMP prediction. During training, RISE identifies potentially unreliable supervision, estimates sample-specific structural reliability from residue-level pLDDT, and conservatively refines supervision using multiple prediction sources.

## Method Overview

RISE contains three main reliability-aware components:

1. **Neighborhood-based candidate identification**  
   Sequence and structure descriptors are L2-normalized, concatenated, and used to build a cross-sample neighborhood. Samples with strong local disagreement between the observed label and current fused prediction are treated as correction candidates.

2. **Structural reliability calibration**  
   Residue-level ESMFold pLDDT values are summarized using the mean confidence, a lower confidence quantile, and the fraction of low-confidence residues. The resulting sample-level reliability is used to discount structural evidence.

3. **Verified soft supervision refinement**  
   Candidate samples are evaluated using four complementary sources:
   - sequence prediction;
   - neighborhood consensus;
   - historical prediction;
   - reliability-adjusted structural prediction.

   A soft target is accepted only when conservative verification conditions are satisfied. Otherwise, the previous effective supervision is retained.

At inference time, supervision refinement is disabled. Prediction uses the untransformed sequence and structural evidence fused by the evidential prediction backbone.

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

## Data Protocol

The controlled label-noise experiments use predefined train/validation/test splits:

```text
data/<dataset>/noise_<rate>/rep<id>/
├── train.csv
├── val.csv
├── test.csv
├── noise_indices.csv
└── metadata.json
```

The main CSV fields are:

| Column | Description |
|---|---|
| `id` | Sample identifier |
| `sequence` | Peptide sequence |
| `clean_target` | Original clean binary label |
| `target` | Observed label used for training |
| `is_noisy` | Whether `target` differs from `clean_target` |
| `split` | Dataset split when included |

Training uses `target`. Validation and test evaluation use clean labels.

Controlled symmetric label noise is applied **only to the training split**. Validation and test labels remain unchanged.

Noise rates are `0.0`, `0.1`, `0.2`, and `0.3`, with five repetitions.

## Sequence Features

RISE uses frozen ESM2-650M residue-level representations. ESM2 parameters are not updated during RISE training. The extracted residue representations are passed to a one-layer bidirectional LSTM with hidden size 128 in each direction.

## Structure Features

Structures are predicted in advance with ESMFold and stored as PDB files.

Each peptide is converted into a residue graph containing:

- sequential residue edges;
- spatial-contact edges within 8 Å;
- eight spatial nearest-neighbor edges.

The structural encoder uses three residual GraphSAGE layers with hidden size 128.

Residue-level pLDDT is normalized to `[0, 1]` and retained as the first structural node-feature column because the training pipeline uses this confidence profile for sample-specific structural reliability estimation.

## Main Implementation Settings

| Setting | Value |
|---|---:|
| Sequence encoder | 1-layer BiLSTM |
| BiLSTM hidden size | 128 per direction |
| Structural encoder | 3-layer residual GraphSAGE |
| GraphSAGE hidden size | 128 |
| Dropout | 0.30 |
| Structural contact threshold | 8 Å |
| Structural spatial KNN | 8 |
| Cross-sample neighborhood size | 20 |
| Candidate threshold | 0.80 |
| Reliability weights | 0.50 / 0.30 / 0.20 |
| Reliability reference quantiles | 0.15 / 0.70 |
| Historical EMA coefficient | 0.70 |
| Soft-target source weights | 0.35 / 0.35 / 0.15 / 0.15 |
| Pseudo-label confidence threshold | 0.55 |
| Pseudo-label margin threshold | 0.10 |
| Batch size | 16 |
| Backbone learning rate | 3e-4 |
| Transition learning rate | 1e-3 |
| Weight decay | 1e-4 |
| Warm-up | 5 epochs |
| Supervision assessment interval | 5 epochs |
| Early-stopping patience | 40 |

## External Resources

Large external resources are not committed to this repository:

- ESM2-650M model weights;
- ESMFold PDB structures;
- extracted feature caches;
- trained checkpoints and runtime logs.

Create a local path configuration:

```bash
cp configs/paths.example.yaml configs/paths.yaml
```

Then edit the local file, for example:

```yaml
runtime:
  device: cuda:0
  esm_model_path: /path/to/esm2_650M
  output_root: /path/to/rise_outputs
  cache_root: /path/to/rise_cache

datasets:
  xuamp:
    root: ./data/XUAMP
    pdb_dir: /path/to/xuamp/esmfold_pdb

  genpept:
    root: ./data/GenPept-Curated-2025
    pdb_dir: /path/to/genpept/esmfold_pdb
```

`configs/paths.yaml` is intended to remain local and should not be committed.

## Installation

```bash
conda create -n rise python=3.10 -y
conda activate rise
pip install -r requirements.txt
```

The repository currently specifies:

```text
numpy==1.26.4
pandas==2.3.3
torch==2.5.1
scikit-learn==1.7.2
transformers==4.44.2
PyYAML==6.0.3
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

After training, evaluate the best validation checkpoint with:

### XUAMP

```bash
CUDA_VISIBLE_DEVICES=0 python test.py \
  --config configs/xuamp.yaml \
  --paths configs/paths.yaml \
  --noise 0.100 \
  --rep 1
```

### GenPept-Curated-2025

```bash
CUDA_VISIBLE_DEVICES=0 python test.py \
  --config configs/genpept.yaml \
  --paths configs/paths.yaml \
  --noise 0.100 \
  --rep 1
```

Testing loads `best_model.pt` from the corresponding training run directory.

No candidate identification, structural reliability calibration, pseudo-label construction, or supervision refinement is performed during testing. Final prediction is obtained from the fused sequence--structure evidential prediction.

## Output Files

For XUAMP with noise rate `0.100` and repetition `1`, outputs are organized under:

```text
<output_root>/xuamp/noise_0.100/rep1/
```

Training produces files including:

```text
best_model.pt
run_config.json
val_metrics.json
val_predictions.csv
train_history.csv
train_info.csv
split_indices.npz
```

Testing additionally produces:

```text
test_metrics.json
test_predictions.csv
```

## Regenerating Controlled Label Noise

```bash
python data_preparation/generate_symmetric_noise.py --help
```

The controlled-noise generation protocol:

- modifies training labels only;
- keeps validation and test labels clean;
- applies class-balanced symmetric label flipping;
- generates the predefined noisy training files used by the experiments.

## Main Results

Results are reported as mean ± standard deviation over five repetitions.

| Dataset | Clean ACC | Clean AUC | Clean MCC | MCC@10% | MCC@20% | MCC@30% |
|---|---:|---:|---:|---:|---:|---:|
| XUAMP | 0.750 ± 0.009 | 0.803 ± 0.005 | 0.531 ± 0.012 | 0.503 ± 0.007 | 0.485 ± 0.011 | 0.466 ± 0.024 |
| GenPept-Curated-2025 | 0.930 ± 0.002 | 0.974 ± 0.002 | 0.863 ± 0.004 | 0.847 ± 0.005 | 0.829 ± 0.005 | 0.753 ± 0.006 |

These values correspond to the manuscript results. If implementation details, dependencies, data preprocessing, or random seeds are changed, the reported results should be revalidated.

## Citation

```bibtex
@article{liu2026rise,
  title  = {Rethinking Sequence--Structure Learning from Unreliable Evidence for Antimicrobial Peptide Prediction},
  author = {Liu, Fei and Jiang, Shouwei and Wu, Le and Lu, Wenjie and Wang, Feilong and Wu, Guangzhou and Wu, Han and Ji, Shengwei and Hong, Richang},
  year   = {2026},
  note   = {Manuscript}
}
```

## License

See [LICENSE](LICENSE).

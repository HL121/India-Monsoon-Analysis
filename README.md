# Monsoon Analysis

This repository contains the climate-index analysis pipeline used to evaluate
Prithvi-WxC latent representations against boreal-summer monsoon and tropical
variability indices. The pipeline trains sequence models on extracted encoder
and decoder latents and reports leave-one-year-out prediction skill for MISO,
MJO, and IOD targets.

## Repository Layout

- `src/`: reusable Python pipeline for data loading, model definition, training,
  evaluation, and plotting.
- `configs/`: YAML experiment settings for MISO, MJO, and IOD.
- `climate_indices_data/`: observed climate-index time series used as targets.
- `outputs/`: generated metric summaries and figures.
- `extract_latent.ipynb`: notebook used inside a Prithvi-WxC checkout to produce
  the latent vectors consumed by this repository.

## Installation

Create an environment for the climate-index analysis code:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The dependencies for `extract_latent.ipynb` are managed separately by the
Prithvi-WxC repository and are not included in this requirements file.

## Latent Extraction

The analysis pipeline assumes daily latent vectors are available as:

```text
latent_output/<year>/encoder_<timestamp>.npy
latent_output/<year>/decoder_<timestamp>.npy
```

To generate these files, clone and install Prithvi-WxC:

```bash
git clone https://github.com/NASA-IMPACT/Prithvi-WxC.git
cd Prithvi-WxC
pip install '.[examples]'
```

Create a directory name: `india_monsoon_prithvi`. Place `extract_latent.ipynb` in:

```text
Prithvi-WxC/india_monsoon_prithvi/extract_latent.ipynb
```

Start Jupyter from `Prithvi-WxC/india_monsoon_prithvi` or otherwise ensure the
notebook kernel uses that folder as its working directory. The notebook downloads
the required Prithvi-WxC weights and climatology files, extracts regional
encoder and decoder features, and writes them to `latent_output/<year>/`.
Place this `latent_output/` folder in the root of this repository before running
the experiments.

## Running Experiments

From the repository root, run:

```bash
python -m src.run_miso_experiment
python -m src.run_mjo_experiment
python -m src.run_iod_experiment
```

Each run reads its corresponding file in `configs/`, loads the latent vectors
and observed index targets, trains LSTM, Transformer, and joint compressor plus
Transformer models, and writes outputs to the configured folders under
`outputs/`.

Sequence length, years, target columns, model hyperparameters, and output paths
are controlled in the YAML files. To reproduce alternate sequence-length
experiments, update `sequence.seq_len` in the relevant config and rerun the
corresponding module.

## Outputs

Metric summaries are saved as CSV files in `outputs/metrics/`. Figures are saved
under `outputs/figures/`, including full-period observed-versus-predicted time
series and, for bivariate MISO/MJO targets, phase-space trajectory plots.

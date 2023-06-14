# Adapting Protein Language Models for Rapid DTI Prediction

This repository documents the code used to generate the results for our [PNAS](https://www.pnas.org/doi/10.1073/pnas.2220778120) article. The updated package, which is continuously being developed, can be found at [this repository](https://github.com/samsledje/ConPLex). Please submit an issue or email samsl@mit.edu with any questions.

### Sample Usage

`python train_DTI.py --exp-id ExperimentName --config configs/default_config.yaml`

### Repository Organization

- `src`: Python files containing protein and molecular featurizers, prediction architectures, and data loading
- `scripts`: Bash files to run benchmarking tasks
  - `CMD_BENCHMARK_DAVIS.sh` -- Run DTI classification benchmarks on DAVIS data set. Can be easily modified for other classification data sets
  - `CMD_BENCHMARK_TDC_DTI_DG.sh` -- Run benchmarks for [TDC](https://tdcommons.ai) [DTI-DG](https://tdcommons.ai/benchmark/dti_dg_group/bindingdb_patent/) regression task
  - `CMD_BENCHMARK_DUDE_CROSSTYPE.sh` -- Evaluate trained model on [DUDe](http://dude.docking.org) decoy performance for kinase and GPCR targets
  - `CMD_BENCHMARK_DUDE_WITHINTYPE.sh` -- The same as above, but with half of kinase, gpcr, protease, and nuclear targets
- `models`: Pre-trained protein language models
- `dataset`: Data sets to benchmark on, most are from [MolTrans](https://academic.oup.com/bioinformatics/article/37/6/830/5929692)
  - `DAVIS`
  - `BindingDB`
  - `BIOSNAP`
  - `DUDe` 
- `nb`: Jupyter notebooks for data generation and exploration
- `train_DTI.py` -- Main training script to run DTI classification benchmarks
- `DUDE_evaluate_decoys.py` -- Compare predictions of a trained model between a target and known true binders/decoys. Visualize embedding space
- `DUDE_summarize_decoys.py` -- Given a directory of protein targets, summarize active/decoy discriminative performance by target type


### Reference

- Described in [our PNAS paper](https://www.pnas.org/doi/10.1073/pnas.2220778120)
- Previously appeared in [NeurIPS MLSB 2021](https://www.mlsb.io/papers_2021/MLSB2021_Adapting_protein_language_models.pdf) and [NeurIPS MLSB 2022](https://www.biorxiv.org/content/10.1101/2022.11.03.515086v1), and on [bioRxiv](https://www.biorxiv.org/content/10.1101/2022.12.06.519374v1).

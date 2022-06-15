# Adapting Protein Language Models for Rapid DTI Prediction

This repository is a work in progress. Please submit an issue or email samsl@mit.edu with any questions.

### Sample Usage

`python train_plm_dti.py --replicate 1 --wandb-proj Default_Project --task davis --model-type SimpleCosine --mol-feat Morgan_f --prot-feat BeplerBerger_DSCRIPT_cat_f --exp-id Davis_SimpleCosine_Morgan_BeplerBerger_DSCRIPT_cat_rep1`

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
- `train_plm_dti.py` -- Main training script to run DTI classification benchmarks, with options for task, featurizer, architecture, logging, etc.
- `train_plm_dti-TDC-DG.py` -- Run TDC DTI-DG benchmarking with options for featurizer, model architecture, logging, etc.
- `train_plm_dti-DUDE.py` -- Main training script, modified to additionally train with a triplet distance loss on the DUDe decoy training set
- `train_plm_dti-{contrastive, CELoss}` -- Main training script, modified to additionally train with a {CosineEmbeddingLoss, TripletDistanceLoss} on the benchmark task data set
- `DUDE_evaluate_decoys.py` -- Compare predictions of a trained model between a target and known true binders/decoys. Visualize embedding space
- `DUDE_summarize_decoys.py` -- Given a directory of protein targets, summarize active/decoy discriminative performance by target type


### Reference

- Described in [NeurIPS MLSB 2021](https://www.mlsb.io/papers_2021/MLSB2021_Adapting_protein_language_models.pdf)
- Based on code from [MolTrans](https://github.com/kexinhuang12345/MolTrans)

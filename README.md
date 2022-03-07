# Adapting Protein Language Models for Rapid DTI Prediction

This repository is a work in progress. Please submit an issue or email samsl@mit.edu with any questions.

### Sample Usage

`python train_plm_dti.py --replicate 1 --wandb-proj Default_Project --task davis --model-type SimpleCosine --mol-feat Morgan_f --prot-feat BeplerBerger_DSCRIPT_cat_f --exp-id Davis_SimpleCosine_Morgan_BeplerBerger_DSCRIPT_cat_rep1`

### Reference

- Described in [NeurIPS MLSB 2021](https://www.mlsb.io/papers_2021/MLSB2021_Adapting_protein_language_models.pdf)
- Based on code from [MolTrans](https://github.com/kexinhuang12345/MolTrans)

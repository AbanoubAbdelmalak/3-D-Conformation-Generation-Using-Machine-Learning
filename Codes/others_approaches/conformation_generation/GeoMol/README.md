# GeoMol: Torsional Geometric Generation of Molecular 3D Conformer Ensembles
This page contains my trials to make changes to the code from [GeoMol](https://github.com/PattanaikL/GeoMol) to run on newer PyTorch-Geometric versions
---
This repository contains a method to generate 3D conformer ensembles directly from the molecular graph as described in
our [paper](https://arxiv.org/pdf/2106.07802.pdf). 



## Usage
This should result in two different directories, one for each half of GEOM. You should place the qm9 conformers directory
in the `data/QM9/` directory and do the same for the drugs directory. This is all you need to train the model:

`python train.py --data_dir data/QM9/qm9/ --split_path data/QM9/splits/split0.npy --log_dir ./test_run --n_epochs 250 --dataset qm9`

Use the provided script to generate conformers. The `test_csv` arg should be a csv file with SMILES in the first column,
and the number of conformers you want to generate in the second column. This will output a compressed dictionary of rdkit
mols in the `trained_model_dir` directory (unless you provide the `out` arg):

`python generate_confs.py --trained_model_dir trained_models/qm9/ --test_csv data/QM9/test_smiles.csv --dataset qm9`

However, note that to reproduce the numbers in our paper, one needs additionally to run scripts/clean_smiles.py to account for inconsistent molecules in the dataset. See also count_geomol_failures.ipynb .
You can use the provided `visualize_confs.ipynb` jupyter notebook to visualize the generated conformers.

## Additional comments



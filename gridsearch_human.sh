#!/bin/bash
# Edit the following 3 lines according to your setup:
SPARSECHEM_PATH=/home/esztergu/git/SparseChem/examples/chembl/
CHEMBL_PATH=~/git/chembl-pipeline/output/chembl_29/
conda activate new_SparseChem
export LD_LIBRARY_PATH=/home/esztergu/miniconda3/envs/new_SparseChem/lib/:$LD_LIBRARY_PATH

for do in 0.5 0.6 0.7 0.8 0.9
do
    for wd in 1e-4 1e-5 1e-6 1e-7 0
    do
        for hidden in 1600 1800 2000 2500 3000 3500 4000 6000 8000 10000 15000
        do
            python ${SPARSECHEM_PATH}train.py --x ${CHEMBL_PATH}chembl_29_X_only_human.npy --y ${CHEMBL_PATH}chembl_29_thresh_only_human.npy --folding ${CHEMBL_PATH}folding_only_human.npy --fold_va 3 --fold_te 4 --hidden_sizes $hidden --weight_decay $wd --dropouts_trunk $do --internal_batch_max 1000 --output_dir human_only
        done
    done
done

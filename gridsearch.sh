#!/bin/bash
export LD_LIBRARY_PATH=/home/esztergu/miniconda3/envs/new_SparseChem/lib/:$LD_LIBRARY_PATH
conda activate new_SparseChem
for do in 0.5 0.6 0.7 0.8
do
    for wd in 0 1e-1 1e-2 1e-3 1e-4 1e-5 1e-6 1e-7 
    do
        for hidden in 3000 4000 6000 8000 10000 15000
        do
            python /home/esztergu/git/SparseChem/examples/chembl/train.py --x ~/git/chembl-pipeline/output/chembl_29/chembl_29_X_new.npy --y ~/git/chembl-pipeline/output/chembl_29/chembl_29_thresh_new.npy --folding ~/git/chembl-pipeline/output/chembl_29/folding_new.npy --fold_va 3 --fold_te 4 --hidden_sizes $hidden --weight_decay $wd --dropouts_trunk $do --weights_class ~/git/chembl-pipeline/output/chembl_29/agg_weight.csv --internal_batch_max 2000 --output_dir rodent
        done
    done
done

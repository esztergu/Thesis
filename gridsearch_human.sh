#!/bin/bash
export LD_LIBRARY_PATH=/home/esztergu/miniconda3/envs/new_SparseChem/lib/:$LD_LIBRARY_PATH
conda activate new_SparseChem
for do in 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9
do
    for wd in  0 1e-1 1e-2 1e-3 1e-4 1e-5 1e-6 1e-7 
    do
        for hidden in  100 200 300 400 500 600 700 800 900 1000 1200 1400 1600 1800 2000 2500 3000 3500 4000 8000 10000 15000 20000 
        do
            python /home/esztergu/git/SparseChem/examples/chembl/train.py --x ~/git/chembl-pipeline/output/chembl_29/chembl_29_X_only_human.npy --y ~/git/chembl-pipeline/output/chembl_29/chembl_29_thresh_only_human.npy --folding ~/git/chembl-pipeline/output/chembl_29/folding_only_human.npy --fold_va 3 --fold_te 4 --hidden_sizes $hidden --weight_decay $wd --dropouts_trunk $do --internal_batch_max 1000
        done
    done
done

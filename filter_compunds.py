import numpy as np
import pandas as pd
import scipy as sp
import sparsechem as sc
chembl_path="dependencies/EszterGulyasBSc/chembl-pipeline-main/output/chembl_29/"

# Opening the required files
folding = np.load(chembl_path+"folding.npy")
y = np.load(chembl_path+"chembl_29_thresh.npy",allow_pickle=True).item(0)
targets = pd.read_csv(chembl_path+"chembl_29_targets.csv",header=None)
targets_h = pd.read_csv(chembl_path+"chembl_29_targets_homos.csv",header=None)

targets_h["is_homo"] = 1

joined = targets.join(targets_h.set_index(0), on=0, how="outer")
is_homo = np.array(~joined['is_homo'].isnull()) # Mask which is True for homo sapiens targets
mask = np.abs(y) # y: -1: inactive, 1: active, 0: unmeasured -> mask: 1: measured, 0: unmeasured
# Repeating 4 times for the thresholds
is_homo = np.repeat(is_homo,4)
homo_count = mask @ is_homo
is_only_rodent = (homo_count == 0)
folding_3_or_4 = (folding == 3) | (folding == 4)
keep = ~(is_only_rodent & folding_3_or_4)
keep_only_human = ~(is_only_rodent)

y_csr = y.tocsr()
y_new = y_csr[keep]
folding_new = folding[keep]

# Saving the filtered results
np.save(chembl_path+"folding_new.npy",folding_new)
np.save(chembl_path+"chembl_29_thresh_new.npy",y_new)
x = np.load(chembl_path+"chembl_29_X.npy",allow_pickle=True).item(0)
x_new = x[keep]
np.save(chembl_path+"chembl_29_X_new.npy",x_new)

n=5
fold_pos, fold_neg = sc.class_fold_counts(y_new, folding_new)
aggregation_weight = ((fold_pos >= n).all(0) & (fold_neg >= n)).all(0).astype(np.float64)

np.save(chembl_path+"agg_weight.npy",aggregation_weight * is_homo)

data = {
  "task_type": "OTHER",
  "training_weight": 1.0,
  "aggregation_weight": aggregation_weight * is_homo
}

df = pd.DataFrame(data)
df.index.name = "task_id"
df.to_csv(chembl_path+"agg_weight.csv")

# Creating the only human versions
y_only_human = y_csr[keep_only_human]
y_only_human = y_only_human.tocsc()[:,is_homo]
folding_only_human = folding[keep_only_human]

np.save(chembl_path+"folding_only_human.npy",folding_only_human)
np.save(chembl_path+"chembl_29_thresh_only_human.npy",y_only_human)

x_only_human = x[keep_only_human]
np.save(chembl_path+"chembl_29_X_only_human.npy",x_only_human)
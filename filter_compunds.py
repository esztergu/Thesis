import numpy as np
import pandas as pd
import scipy as sp
import sparsechem as sc
chembl_path="dependencies/EszterGulyasBSc/chembl-pipeline-main/output/chembl_29/"

folding = np.load(chembl_path+"folding.npy")
y = np.load(chembl_path+"chembl_29_thresh.npy",allow_pickle=True).item(0)
# <1443601x10296 sparse matrix of type '<class 'numpy.float64'>'with 31167071 stored elements in COOrdinate format>
targets = pd.read_csv(chembl_path+"chembl_29_targets.csv",header=None) # 2574
targets_h = pd.read_csv(chembl_path+"chembl_29_targets_homos.csv",header=None) #2526

targets_h["is_homo"] = 1

joined = targets.join(targets_h.set_index(0), on=0, how="outer")
is_homo = np.array(~joined['is_homo'].isnull()) # True ahol homo, 48 False
mask = np.abs(y) # -1: inaktiv 1: aktiv 0: nincs lemérve; ebből kell 1: inaktiv vagy aktiv 0: nincs lemérve
# Repeating 4 times for the thresholds
is_homo= np.repeat(is_homo,4) # matrix korrekció
homo_count = mask @ is_homo # matrix szorzás
# homo_count.shape
# sum(homo_count == 0) 4743 csak egér
# print(sum(homo_count == 0))
is_only_rodent = (homo_count == 0)
folding_3_or_4 = (folding == 3) | (folding == 4)
keep = ~(is_only_rodent & folding_3_or_4) # nem csak rodent vagy nem a folding 3 és 4-ben van
# sum(keep) 1441631
# keep.shape (1443601,) 1970 vegyületet dobtunk el, ez marad a folding 3 és 4-re, ami jó, sok maradt tanulni
keep_only_human = ~(is_only_rodent)

y_csr = y.tocsr()
# y_csr
# <1443601x10296 sparse matrix of type '<class 'numpy.float64'>' with 31167071 stored elements in Compressed Sparse Row format>
y_new = y_csr[keep]
# <1441631x10296 sparse matrix of type '<class 'numpy.float64'>' with 31157117 stored elements in Compressed Sparse Row format>
folding_new = folding[keep]
# folding_new.shape (1441631,)

np.save(chembl_path+"folding_new.npy",folding_new)
np.save(chembl_path+"chembl_29_thresh_new.npy",y_new)
x = np.load(chembl_path+"chembl_29_X.npy",allow_pickle=True).item(0)
# x.shape (1443601, 32000)
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
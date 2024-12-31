import sparsechem as sc
import os
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import sqlite3
#sc_path="/home/esztergu/git/SparseChem/examples/chembl/"
sc_path="/home/esztergu/git/Thesis/"

print ("hidden, dropout, weight_decay, auc")

#res_human = sc.load_results(sc_path + "models/sc_run_h10000_ldo_r_ldo_t0.7_wd1e-05_lr0.001_lrsteps10_ep20_fva3_fte4.json") #auc=0.811570
res_human = sc.load_results(sc_path + "human_only/sc_run_h10000_ldo_r_ldo_t0.7_wd1e-05_lr0.001_lrsteps10_ep20_fva3_fte4.json") #au=0.811570

my_hidden_size = res_human["conf"].hidden_sizes[0]
my_dropout = res_human["conf"].dropouts_trunk[0]
my_weight_decay = res_human["conf"].weight_decay
my_auc_score = res_human["validation"]["classification_agg"].roc_auc_score
human_auc=my_auc_score
print (f"Human model: {my_hidden_size}, {my_dropout}, {my_weight_decay}, {my_auc_score}")


#res_rodent = sc.load_results(sc_path + "rodent/sc_run_h6000_ldo_r_ldo_t0.8_wd1e-05_lr0.001_lrsteps10_ep20_fva3_fte4.json") #auc=0.801972
res_multispecies = sc.load_results(sc_path + "multispecies/sc_run_h6000_ldo_r_ldo_t0.8_wd1e-05_lr0.001_lrsteps10_ep20_fva3_fte4.json") #auc=0.801972

my_hidden_size = res_multispecies["conf"].hidden_sizes[0]
my_dropout = res_multispecies["conf"].dropouts_trunk[0]
my_weight_decay = res_multispecies["conf"].weight_decay
my_auc_score = res_multispecies["validation"]["classification_agg"].roc_auc_score

print (f"Multispecies model: {my_hidden_size}, {my_dropout}, {my_weight_decay}, {my_auc_score}")

auc_human=res_human["validation"]["classification"].roc_auc_score
auc_multispecies=res_multispecies["validation"]["classification"].roc_auc_score


folding=np.load("/home/esztergu/git/chembl-pipeline/output/chembl_29/folding_only_human.npy")
y_human=np.load("/home/esztergu/git/chembl-pipeline/output/chembl_29/chembl_29_thresh_only_human.npy", allow_pickle=True).item(0)
y_multispecies=np.load("/home/esztergu/git/chembl-pipeline/output/chembl_29/chembl_29_thresh_new.npy", allow_pickle=True).item(0)

# Selecting all human targets with at least 5 positive and negative samples in all folds (This criteria in SparseChem to compute AUC)
n=5
fold_pos, fold_neg = sc.class_fold_counts(y_human.tocsr(), folding)
aggregation_weight = ((fold_pos >= n).all(0) & (fold_neg >= n)).all(0).astype(np.float64)
assert np.abs((aggregation_weight*auc_human).sum()/aggregation_weight.sum()-human_auc) < 1e-4, "Computed aggregation weight can not reconstruct model AUC" # 0.811570

auc_human_filtered=auc_human[aggregation_weight==1.0].values

# Matching targets by their Chembl ID
target_all=pd.read_csv("/home/esztergu/git/chembl-pipeline/output/chembl_29/chembl_29_targets.csv", header=None)
target_homo=pd.read_csv("/home/esztergu/git/chembl-pipeline/output/chembl_29/chembl_29_targets_homos.csv", header=None)

target_all_4=np.repeat(target_all.values,4,axis=0)
target_homo_4=np.repeat(target_homo.values,4,axis=0)

#target_all_filtered=target_all_4[df.aggregation_weight==1.0]
target_homo_filtered=target_homo_4[aggregation_weight==1.0]

# set(target_homo.values.flatten()).difference(target_all.values.flatten()) ez set() szóval jó
# set(target_all_filtered.flatten()).difference(target_homo_filtered.flatten())

target_all_4_thr=target_all_4.flatten()+np.array(["_0","_1","_2","_3"]*target_all.shape[0])
target_homo_4_thr=target_homo_4.flatten()+np.array(["_0","_1","_2","_3"]*target_homo.shape[0])

selected_targets=set(target_homo_4_thr[aggregation_weight==1.0])

all_mask=[(t in selected_targets) for t in list (target_all_4_thr)]
auc_multispecies_filtered=auc_multispecies[all_mask].values #mean-je 0.8112741840355688

all_names=target_all_4_thr[all_mask]
homo_names=target_homo_4_thr[aggregation_weight==1.0]
assert (all_names==homo_names).all(), "The two set of targets are different"


dif=auc_multispecies_filtered-auc_human_filtered
#dif.argmax()
# 138
# all_names[138]
# 'CHEMBL1628481_1'
top10=dif.argsort()[-10:]
top30=dif.argsort()[-30:]

plt.cla()
plt.scatter(auc_human_filtered,auc_multispecies_filtered,s=1)
plt.scatter(auc_human_filtered[top10],auc_multispecies_filtered[top10],c="red",s=2)
plt.plot([0,1],[0,1],c="red")
plt.savefig("target_compare_top10.pdf")

targets=[]
top10_targets=all_names[top10]
for target in top10_targets:
    print(target)
    targets.append(target[:-2])
similars=pd.read_csv("Similar_targets.csv")

targets_30=[]
top30_targets=all_names[top30]
for target in top30_targets:
    print(target)
    targets.append(target[:-2])
#30-ban sincsen

# CHEMBL2123 Pyrimidinergic receptor P2Y4
# CHEMBL3217390 Transcription initiation factor TFIID subunit 1
# CHEMBL3429 Estrogen-related receptor alpha
# CHEMBL5542 DNA polymerase eta
# CHEMBL4941 S-methyl-5-thioadenosine phosphorylase
# CHEMBL4071 Cathepsin G
# CHEMBL612547 HFL1
# CHEMBL1615321 microRNA 30a
# CHEMBL4338 Purine nucleoside phosphorylase
# CHEMBL1615381 Menin

# CHEMBL2123_0
# CHEMBL3217390_0
# CHEMBL3429_0
# CHEMBL5542_1
# CHEMBL4941_2
# CHEMBL4071_1
# CHEMBL612547_0
# CHEMBL1615321_0
# CHEMBL4338_0
# CHEMBL1615381_2


#rodent_targets=set(target_all.values.flatten()).difference(target_homo.values.flatten())
#df_rodent_targets=pd.DataFrame(rodent_targets)
#df_rodent_targets.to_csv("rodent_targets.csv",index=False)


# egér humán páros
# git-be feltenni

#conn = sqlite3.connect("/home/esztergu/git/chembl-pipeline/input/chembl_29_sqlite/chembl_29.db")
#df_prefname=pd.read_sql_query("SELECT PREF_NAME,ORGANISM FROM TARGET_DICTIONARY WHERE CHEMBL_ID = ?", conn, params=("CHEMBL1615381",))
# df=pd.read_sql("SELECT column_name FROM table_name WHERE column_name2 = %s", con=engine, params=(variable_name,))
#target_dictionary=pd.read_sql_query("select * from target_dictionary", conn)
#conn.close()
#just_rodent=target_dictionary.join(df_rodent_targets.set_index(0), on="chembl_id", how="inner")



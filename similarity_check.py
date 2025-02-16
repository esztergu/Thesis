import sparsechem as sc
import torch
import pandas as pd
import numpy as np
import sqlite3

#export LD_LIBRARY_PATH=/home/esztergu/miniconda3/envs/new_SparseChem/lib/:$LD_LIBRARY_PATH

# Best model:
# 8000      0.9  1.000000e-05  0.800169

state_dict=torch.load("multispecies/sc_run_h8000_ldo_r_ldo_t0.9_wd1e-05_lr0.001_lrsteps10_ep20_fva3_fte4.pt")
w=state_dict['classLast.net.initial_layer.2.weight'].cpu()

target_all=pd.read_csv("dependencies/EszterGulyasBSc/chembl-pipeline-main/output/chembl_29/chembl_29_targets.csv", header=None)
target_homo=pd.read_csv("dependencies/EszterGulyasBSc/chembl-pipeline-main/output/chembl_29/chembl_29_targets_homos.csv", header=None)

target_all_4=np.repeat(target_all.values,4,axis=0)
#target_homo_4=np.repeat(target_homo.values,4,axis=0)

target_all_4_thr=target_all_4.flatten()+np.array(["_0","_1","_2","_3"]*target_all.shape[0])
#target_homo_4_thr=target_homo_4.flatten()+np.array(["_0","_1","_2","_3"]*target_homo.shape[0])

rodent_targets=set(target_all.values.flatten()).difference(target_homo.values.flatten())

def targets_sim(id_h, thr_h, id_r, eps=1e-15):
    max_sim=0
    best_r= None
    i=id_h*4+thr_h

    for j in range(id_r*4, (id_r+1)*4):
        denom=np.linalg.norm(w[i])*np.linalg.norm(w[j])
        if denom < eps:
            continue
        tmp=np.dot(w[i],w[j])/denom
        #print (f"{j} -> {tmp}")
        if tmp > max_sim:
            max_sim = tmp
            best_r = j

    #print (f"{best_r} -> {max_sim}")
    return max_sim

target_all_list=target_all[0].tolist()
rodent_index=[i for i in range(len(target_all_list)) if target_all_list[i] in rodent_targets]

top10_t_d=pd.read_csv("top10_targets_details.csv") # This maybe less than 10 because of repeated apperance, which is fine
top10 = np.load("top10_targets.npy", allow_pickle=True)

top10_thr=[x[-1] for x in top10]
top10_cid=[x[:-2] for x in top10]

homo_index={}
for i in range(len(target_all_list)):
    homo_index[target_all_list[i]]=i


# homo_index=[i for i in range(len(target_all_list)) if target_all_list[i] in top10_cid]

conn = sqlite3.connect("dependencies/EszterGulyasBSc/chembl-pipeline-main/input/chembl_29_sqlite/chembl_29.db")


for i in range(len(top10_cid)):
    max_sim=0
    best_rodent=None
    for id_2 in rodent_index:
        sim=targets_sim(homo_index[top10_cid[i]],int(top10_thr[i]), id_2)
        if sim > max_sim:
            max_sim = sim
            best_rodent = id_2

    df_prefname_1=pd.read_sql_query("SELECT PREF_NAME,ORGANISM FROM TARGET_DICTIONARY WHERE CHEMBL_ID = ?", conn, params=(target_all_list[homo_index[top10_cid[i]]],))

    if best_rodent is None:
        #print(f"No pair for: {target_all_list[id_1]}({id_1})" )
        pass
    else:
        df_prefname_2=pd.read_sql_query("SELECT PREF_NAME,ORGANISM FROM TARGET_DICTIONARY WHERE CHEMBL_ID = ?", conn, params=(target_all_list[best_rodent],))
        #print(f"Best pair: {target_all_list[id_1]}({id_1})({df_prefname_1['pref_name'][0]}), {target_all_list[best_homo]}({best_homo})({df_prefname_2['pref_name'][0]}), {max_sim}" )
        print(f"{target_all_list[homo_index[top10_cid[i]]]}_{top10_thr[i]},\"{df_prefname_1['pref_name'][0]}\", {target_all_list[best_rodent]},\"{df_prefname_2['pref_name'][0]}\", {max_sim}" )

# csv, majd latex table

conn.close()
import sparsechem as sc
import torch
import pandas as pd
import numpy as np
import sqlite3

#export LD_LIBRARY_PATH=/home/esztergu/miniconda3/envs/new_SparseChem/lib/:$LD_LIBRARY_PATH
sc_path="/home/esztergu/git/Thesis/"

# Loading the SparseChem model
state_dict=torch.load(sc_path + "multispecies/sc_run_h6000_ldo_r_ldo_t0.8_wd1e-05_lr0.001_lrsteps10_ep20_fva3_fte4.pt")
# Copying the last layer weights
w=state_dict['classLast.net.initial_layer.2.weight'].cpu()

target_all=pd.read_csv("/home/esztergu/git/chembl-pipeline/output/chembl_29/chembl_29_targets.csv", header=None)
target_homo=pd.read_csv("/home/esztergu/git/chembl-pipeline/output/chembl_29/chembl_29_targets_homos.csv", header=None)

target_all_4=np.repeat(target_all.values,4,axis=0)
#target_homo_4=np.repeat(target_homo.values,4,axis=0)

target_all_4_thr=target_all_4.flatten()+np.array(["_0","_1","_2","_3"]*target_all.shape[0])
#target_homo_4_thr=target_homo_4.flatten()+np.array(["_0","_1","_2","_3"]*target_homo.shape[0])

rodent_targets=set(target_all.values.flatten()).difference(target_homo.values.flatten())

#id_1=0 #first target to compare
#id_2=1

def targets_sim(id_1, id_2, eps=1e-15):
    '''This function computes the similarity between two targets by taking the most similiar threshold pair'''
    max_sim=0
    best_i= None
    best_j= None

    for i in range(id_1*4, (id_1+1)*4):
        for j in range(id_2*4, (id_2+1)*4):
            denom=np.linalg.norm(w[i])*np.linalg.norm(w[j])
            if denom < eps:
                continue
            tmp=np.dot(w[i],w[j])/denom
            #print (f"{i}, {j} -> {tmp}")
            if tmp > max_sim:
                max_sim = tmp
                best_i = i
                best_j = j

    #print (f"{best_i}, {best_j} -> {max_sim}")
    return max_sim

target_all_list=target_all[0].tolist()
rodent_index=[i for i in range(len(target_all_list)) if target_all_list[i] in rodent_targets]

target_homo_list=target_homo[0].tolist()
homo_index=[i for i in range(len(target_all_list)) if target_all_list[i] in target_homo_list]

conn = sqlite3.connect("/home/esztergu/git/chembl-pipeline/input/chembl_29_sqlite/chembl_29.db")

print("Murine id,Murine name,Human id,Human name,similarity")

for id_1 in rodent_index:
    
    max_sim=0
    best_homo=None
    for id_2 in homo_index:
        sim=targets_sim(id_1, id_2)
        if sim > max_sim:
            max_sim = sim
            best_homo = id_2

    df_prefname_1=pd.read_sql_query("SELECT PREF_NAME,ORGANISM FROM TARGET_DICTIONARY WHERE CHEMBL_ID = ?", conn, params=(target_all_list[id_1],))

    if best_homo is None:
        #print(f"No pair for: {target_all_list[id_1]}({id_1})" )
        pass
    else:
        df_prefname_2=pd.read_sql_query("SELECT PREF_NAME,ORGANISM FROM TARGET_DICTIONARY WHERE CHEMBL_ID = ?", conn, params=(target_all_list[best_homo],))
        #print(f"Best pair: {target_all_list[id_1]}({id_1})({df_prefname_1['pref_name'][0]}), {target_all_list[best_homo]}({best_homo})({df_prefname_2['pref_name'][0]}), {max_sim}" )
        print(f"{target_all_list[id_1]},\"{df_prefname_1['pref_name'][0]}\", {target_all_list[best_homo]},\"{df_prefname_2['pref_name'][0]}\", {max_sim}" )

# csv, majd latex table

conn.close()

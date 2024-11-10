import sparsechem as sc
import torch
import pandas as pd
import numpy as np

state_dict=torch.load("/home/esztergu/git/SparseChem/examples/chembl/rodent/sc_run_h6000_ldo_r_ldo_t0.8_wd1e-05_lr0.001_lrsteps10_ep20_fva3_fte4.pt")
w=state_dict['classLast.net.initial_layer.2.weight'].cpu()

target_all=pd.read_csv("/home/esztergu/git/chembl-pipeline/output/chembl_29/chembl_29_targets.csv", header=None)
target_homo=pd.read_csv("/home/esztergu/git/chembl-pipeline/output/chembl_29/chembl_29_targets_homos.csv", header=None)

target_all_4=np.repeat(target_all.values,4,axis=0)
#target_homo_4=np.repeat(target_homo.values,4,axis=0)

target_all_4_thr=target_all_4.flatten()+np.array(["_0","_1","_2","_3"]*target_all.shape[0])
#target_homo_4_thr=target_homo_4.flatten()+np.array(["_0","_1","_2","_3"]*target_homo.shape[0])

rodent_targets=set(target_all.values.flatten()).difference(target_homo.values.flatten())

id_1=0 #first target to compare
id_2=1

def targets_sim(id_1, id_2):
    max_sim=0
    best_i= None
    best_j= None

    for i in range(id_1*4, (id_1+1)*4):
        for j in range(id_2*4, (id_2+1)*4):
            tmp=np.dot(w[i],w[j])/(np.linalg.norm(w[i])*np.linalg.norm(w[j]))
            #print (f"{i}, {j} -> {tmp}")
            if tmp > max_sim:
                max_sim = tmp
                best_i = i
                best_j = j

    print (f"{best_i}, {best_j} -> {max_sim}")
    return max_sim    
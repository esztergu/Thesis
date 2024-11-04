import sparsechem as sc
import torch
state_dict=torch.load("/home/esztergu/git/SparseChem/examples/chembl/rodent/sc_run_h6000_ldo_r_ldo_t0.8_wd1e-05_lr0.001_lrsteps10_ep20_fva3_fte4.pt")
w=state_dict['classLast.net.initial_layer.2.weight']
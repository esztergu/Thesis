import sparsechem as sc
import os
import pandas as pd
import configargparse

p = configargparse.ArgParser()
p.add("-d", "--dropout", required=False, help="Filter for dropout value", type=float, default=None)
p.add("-w", "--wd", required=False, help="Filter for weight decay", type=float, default=None)
p.add("-m", "--modeldir", required=True, help="Human or human + rodent", type=str, default=None)

options = p.parse_args()

my_files = os.listdir(options.modeldir)
wd = 1e-5
do = 0.9
tmplist = []
#print ("id, hidden, dropout, weight_decay, auc")
for i in my_files:
    if os.path.splitext(i)[1] == ".json":
        res = sc.load_results(options.modeldir + i)
        my_hidden_size = res["conf"].hidden_sizes[0]
        my_dropout = res["conf"].dropouts_trunk[0]
        my_weight_decay = res["conf"].weight_decay
        my_auc_score = res["validation"]["classification_agg"].roc_auc_score
        tmplist.append([my_hidden_size, my_dropout, my_weight_decay, my_auc_score])
#        if my_dropout == do and my_weight_decay == wd:
#            print (f"{i}, {my_hidden_size}, {my_dropout}, {my_weight_decay}, {my_auc_score}")

df = pd.DataFrame(tmplist)
df.columns = ["hidden", "dropout", "weight_decay", "auc"] 
mask = [True] * len(df)
if options.dropout is not None:
    mask &= (df["dropout"]==options.dropout)
if options.wd is not None:
    mask &= (df["weight_decay"]==options.wd)

print(df[mask].sort_values('auc')) 


   


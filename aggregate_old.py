import sparsechem as sc
import os

my_files = os.listdir("/home/esztergu/git/SparseChem/examples/chembl/rodent/")
max_auc = 0
best_dropout = 0
best_hidden_size = 0
best_weight_decay = 0
print ("id, hidden, dropout, weight_decay, auc")
for i in my_files:
    if os.path.splitext(i)[1] == ".json":
        res = sc.load_results("/home/esztergu/git/SparseChem/examples/chembl/rodent/" + i)
        my_hidden_size = res["conf"].hidden_sizes[0]
        my_dropout = res["conf"].dropouts_trunk[0]
        my_weight_decay = res["conf"].weight_decay
        my_auc_score = res["validation"]["classification_agg"].roc_auc_score
        if max_auc < my_auc_score:
            max_auc = my_auc_score
            best_dropout = my_dropout
            best_hidden_size = my_hidden_size
            best_weight_decay = my_weight_decay
        # print (f"id: {i} hidden: {my_hidden_size} dropout: {my_dropout} weight_decay: {my_weight_decay} \t auc: {my_auc_score}")
        print (f"{i}, {my_hidden_size}, {my_dropout}, {my_weight_decay}, {my_auc_score}")

print(f"Best auc value: {max_auc} hidden size: {best_hidden_size} dropout: {best_dropout} weight_decay: {best_weight_decay} ")       


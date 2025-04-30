# Thesis

Download the propriatery dependencies

Unzip it into a new folder named dependencies

```
pip install rdkit
pip install numpy scipy configargparse pandas
pip install cython
```

Clone the SparseChem repository

```
git clone https://github.com/melloddy/SparseChem.git
```

Enter the folder and install the library

```
pip install -e .
```

Enter to the folder of the leader_follower package and install it

```
pip install -e .
```

Copy the create_dataset.py file from this respository into the chembl_pipeline folder by doing so overwriting the original included in the zip

Execute in the chembl_pipeline folder
```
./ run.sh
```

# Running the hyperparameter search

Go back to the root of the present git repository

Execute the following command:
```
python filter_compunds.py
```

Before running the gridsearch.sh and the gridsearch_human.sh make sure to edit the path and the name of the conda environment to be used

Execute the hyperparamter search on the multispecies dataset by running
```
gridsearch.sh
```

This implies training several neural networks on GPU and may take significant amount of time

Execute the hyperparamter search on the human only dataset by running 
```
gridsearch_human.sh
```

This implies training several neural networks on GPU and may take significant amount of time

You can examine the best models for those datasets using the list_params.py script

Change the file names of the best models in the target_compare.py script then execute it


# Selecting the best hyperparameters

Execute the following commands:
```
python list_params.py -m multispecies/
python list_params.py -m human_only/
```

# Analyzing results

Edit target_compare.py in line 14 and 25 to the selected optimal models
Edit similarity_check.py in line 12 to the selected optimal model

Execute the following commands:
```
python target_compare.py
python similarity_check.py
```

# Creating target orthology mapping

Download the following files from OrthoDB (orthodb.org) to the orthoDB folder:
(The Thesis results were created using version 12.)

```
odb12v0_gene_uniprot_only.tab
odb12v0_OG2genes.tab
```

Execute the following commands:
```
python ortho_mapping.py
```

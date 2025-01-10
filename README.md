# Thesis

Download the propriatery dependencies from here: https://owncloud.esat.kuleuven.be/index.php/s/KfwoJRoJa82Ajt2

Unzip it into a new folder named dependencies

pip install rdkit
pip install numpy scipy configargparse pandas
pip install cython

Clone the SparseChem repository

git clone https://github.com/melloddy/SparseChem.git

Enter the folder and install the library

pip install -e .

Enter to the folder of the leader_follower package and install it

pip install -e .

Copy the create_dataset.py file from this respository into the chembl_pipeline folder by doing so overwriting the original included in the zip

Execute run.sh

# Running the hyperparameter search

Before running the gridsearch.sh and the gridsearch_human.sh make sure to edit the path and the name of the conda environment to be used

Execute the hyperparamter search on the multispecies dataset by running the gridsearch.sh
This implies training several neural networks on GPU and may take significant amount of time

Execute the hyperparamter search on the human only dataset by running the gridsearch_human.sh
This implies training several neural networks on GPU and may take significant amount of time

You can examine the best models for those datasets using the list_params.py script

Change the file names of the best models in the target_compare.py script then execute it

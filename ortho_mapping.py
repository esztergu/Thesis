import pandas as pd
import sqlite3
cp_out_path="/home/esztergu/git/Thesis/dependencies/EszterGulyasBSc/chembl-pipeline-main/output/chembl_29/"
conn = sqlite3.connect("/home/esztergu/git/Thesis/dependencies/EszterGulyasBSc/chembl-pipeline-main/input/chembl_29_sqlite/chembl_29.db")
target_all=pd.read_csv(cp_out_path + "chembl_29_targets.csv", header=None)
target_homo=pd.read_csv(cp_out_path + "chembl_29_targets_homos.csv", header=None)

df_list = []
for i in target_all.to_numpy():
    chid= i[0]
    df = pd.read_sql_query(f"SELECT chembl_id, accession FROM target_dictionary as td INNER JOIN target_components AS tc ON tc.tid = td.tid INNER JOIN component_sequences as cs ON cs.component_id = tc.component_id  WHERE chembl_id='{chid}'", conn)
    if len(df) == 1:
        df_list.append(df)

df = pd.concat(df_list)
df_list = None

human_targets=set(target_homo.values.flatten())
rodent_targets=set(target_all.values.flatten()).difference(target_homo.values.flatten())

# Creating ortho group mapping for rodent and human targets
rodent_uni = df[df["chembl_id"].isin(rodent_targets)]
human_uni = df[df["chembl_id"].isin(human_targets)]

orthoUni = pd.read_csv("orthoDB/odb12v0_gene_uniprot_only.tab", sep="\t", header=None)
orthoUni = orthoUni.rename(columns={0:"gene", 1:"accession"})[["gene", "accession"]]

rodent_ortho_mapping = pd.merge(orthoUni, rodent_uni, on="accession")
human_ortho_mapping = pd.merge(orthoUni, human_uni, on="accession")
orthoUni = None # To save memory

OG2genes = pd.read_csv("orthoDB/odb12v0_OG2genes.tab", sep="\t", header=None)
OG2genes = OG2genes.rename(columns={0:"OrthoGroup", 1:"gene"})

rodent_OG_mapping = pd.merge(rodent_ortho_mapping, OG2genes, on="gene")
rodent_OG_mapping = rodent_OG_mapping[["chembl_id", "OrthoGroup"]]
rodent_OG_mapping.to_csv("rodent_OG_mapping.csv")

human_OG_mapping = pd.merge(human_ortho_mapping, OG2genes, on="gene")
OG2genes = None
human_OG_mapping = human_OG_mapping[["chembl_id", "OrthoGroup"]]
human_OG_mapping.to_csv("human_OG_mapping.csv")

rodent_to_human = pd.merge(rodent_OG_mapping, human_OG_mapping, on="OrthoGroup", suffixes=("_rodent", "_human"))
rodent_to_human = rodent_to_human[["chembl_id_rodent", "chembl_id_human"]].drop_duplicates()
rodent_to_human.to_csv("ortholog_mapping.csv")

namemapping=pd.read_sql_query("SELECT PREF_NAME, CHEMBL_ID FROM TARGET_DICTIONARY", conn)
temp=pd.merge(rodent_to_human, namemapping, left_on="chembl_id_rodent", right_on="chembl_id")
temp=temp.rename(columns={"pref_name":"pref_name_rodent"})[["chembl_id_rodent", "chembl_id_human", "pref_name_rodent"]]
rodent_to_human_names=pd.merge(temp, namemapping, left_on="chembl_id_human", right_on="chembl_id").rename(columns={"pref_name":"pref_name_human"})
rodent_to_human_names.to_csv("ortholog_mapping_with_names.csv")

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

print(df)

rodent_targets=set(target_all.values.flatten()).difference(target_homo.values.flatten())
rodent_uni = df[df["chembl_id"].isin(rodent_targets)]

orthoUni = pd.read_csv("/home/aarany/odb12v0_gene_uniprot_only.tab", sep="\t", header=None)
orthoUni = orthoUni.rename(columns={1:"accession"})
ortho_mapping = pd.merge(orthoUni, rodent_uni, on="accession")
orthoUni = None

OG2genes = pd.read_csv("orthoDB/odb12v0_OG2genes.tab", sep="\t", header=None)
OG2genes = OG2genes.rename(columns={1:"gene"})
ortho_mapping = ortho_mapping.rename(columns={0:"gene"})
OG_mapping = pd.merge(ortho_mapping, OG2genes, on="gene")
# OG_mapping = OG_mapping.rename(columns={4:"OrthoGroup"}) ??? hanyadik oszlop
OG_mapping.to_csv("OG_mapping.csv")
OG_mapping = OG_mapping[["chembl_id", "OrthoGroup"]]




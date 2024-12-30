# This file is an extended version of the data generating script published with the paper:
# Simm, J., Arany, A., De Brouwer, E., & Moreau, Y. (2021, October). 
# Expressive graph informer networks. In International Conference on Machine Learning, Optimization, and Data Science (pp. 198-212). 
# Cham: Springer International Publishing.
#
# Dataset filters:
# 1) Select all assays where the organism is "Homo sapiens" or "Mus musculus" and the targe_type is "SINGLE PROTEIN" proteins
# 2) Filter according to the standard_units is "nM" or "ug.mL-1"
# 3) Pick the minimum IC50 for all cells
# 4) Filter credible values: 10^9 > IC50 >= 10^-5
# 5) Refilter proteins that have at least N compounds
# 6) Convert to log scale via pIC50 = 9 - log10(IC50)
# OUT: csv data, compound name, protein name lists, homo sapiens protein name list

import configargparse
import sqlite3
import pandas as pd
import numpy as np
import scipy.io
import os
import logging


p = configargparse.ArgParser(default_config_files=["default.ini"])
p.add('-c', '--config', required=False, is_config_file=True, help='Config file path')
p.add('--sqlite', required=True, type=str, help="ChEMBL sqlite database")
p.add('--mincmpdcount', required=True, help='Minimal number of compounds required for an assays', type=int)
p.add('--thresholds', required=True, help="Thresholds for classification", type=float, action="append")
p.add('--datadir', required=True, help="Data directory to write to (append prefix)", type=str)
p.add('--prefix', required=True, help="Prefix for the current dataset", type=str)
options = p.parse_args()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

outdir = options.datadir + "/" + options.prefix

if not os.path.exists(outdir):
    os.makedirs(outdir)

conn = sqlite3.connect(options.sqlite)

logging.info("Querying sqlite database '%s'" % options.sqlite)
df = pd.read_sql_query("""SELECT molecule_dictionary.chembl_id as cmpd_id, target_dictionary.chembl_id as target_id,  target_dictionary.organism as org,
                          CASE activities.standard_units
                            WHEN 'nM' THEN activities.standard_value
                            WHEN 'ug.mL-1' THEN activities.standard_value / compound_properties.full_mwt * 1E6
                            END ic50,
                          CASE activities.standard_relation 
                            WHEN '<'  THEN '<'
                            WHEN '<=' THEN '<'
                            WHEN '='  THEN '='
                            WHEN '>'  THEN '>'
                            WHEN '>='  THEN '>' 
                            ELSE 'drop' END relation
                          FROM molecule_dictionary 
                          JOIN activities on activities.molregno == molecule_dictionary.molregno 
                          JOIN assays on assays.assay_id == activities.assay_id 
                          JOIN target_dictionary on target_dictionary.tid == assays.tid
                          JOIN compound_properties on compound_properties.molregno = molecule_dictionary.molregno
                          WHERE target_dictionary.organism='Homo sapiens' OR target_dictionary.organism="Mus musculus"
                          AND target_dictionary.target_type='SINGLE PROTEIN' AND
                                activities.standard_type = 'IC50' AND activities.standard_units IN  ('nM','ug.mL-1') AND
                                activities.standard_relation IN ('<', '<=', '=','>', '>=')  AND
                                ic50 < 10e9 AND ic50 >= 10e-5 """, conn)
conn.close()
logging.info("Filtering and thresholding activity data")
# Pick the minimum
df = df.groupby(["target_id","cmpd_id"]).min().reset_index()
# at least N compounds per assay
c  = df.groupby("target_id")["cmpd_id"].nunique()
i  = c[c >= options.mincmpdcount].index
df = df[df.target_id.isin(i)]

df["pic50"] = 9 - np.log10(df["ic50"])
df.to_csv('temporary_continuous.csv')

#Thresholding
value_vars = []
for thr in options.thresholds:
    value_vars.append("%1.1f" % thr)
    thr_str = "%1.1f" % thr
    ## using +1 and -1 for actives and inactives
    df[thr_str] = (df["pic50"] >= thr) * 2.0 - 1.0
    df[thr_str] = np.where(np.logical_and((df["relation"] == '<'), (df['pic50'] < thr)), np.nan, df[thr_str]) 
    df[thr_str] = np.where(np.logical_and((df["relation"] == '>'), (df['pic50'] > thr)), np.nan, df[thr_str]) 

logging.info("Saving data into '%s'" % outdir)
melted = pd.melt(df, id_vars=['target_id','cmpd_id','org'], value_vars=value_vars).dropna()
melted.to_csv('%s/%s_thresh.csv' % (outdir, options.prefix), index = False)

#Write unique compound IDs
np.savetxt("%s/%s_compounds.csv" % (outdir, options.prefix), melted["cmpd_id"].unique(), fmt="%s")
np.savetxt("%s/%s_targets.csv" % (outdir, options.prefix), melted["target_id"].unique(), fmt="%s")
homos=melted[melted.org=="Homo sapiens"]
np.savetxt("%s/%s_targets_homos.csv" % (outdir, options.prefix), homos["target_id"].unique(), fmt="%s")
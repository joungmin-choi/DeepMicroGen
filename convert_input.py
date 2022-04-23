import pandas as pd 
import os
import sys

filename = sys.argv[1]
output_filename = "input_abundance_profiles.csv"

data = pd.read_csv(filename)

phylum_list = []

for i in range(len(data)) :
	phylum_list.append(data["Unnamed: 0"][i].split('|')[1])

data["phylum"] = phylum_list 

phylum_uniq_list = data["phylum"].unique().tolist()
phylum_otu_count_list = []

for i in range(len(phylum_uniq_list)) :
	tmp_phylum = data[data["phylum"] == phylum_uniq_list[i]]
	phylum_otu_count_list.append(len(tmp_phylum))

phylum_otu_count_df = pd.DataFrame({'phylum' : phylum_uniq_list, 'otu_count' : phylum_otu_count_list})
phylum_otu_count_df = phylum_otu_count_df.sort_values(by="otu_count", ascending = False)
phylum_otu_count_df.reset_index(inplace = True, drop = True)

final_data_df = pd.DataFrame()
for i in range(len(phylum_otu_count_df)) :
	tmp_phylum = data[data["phylum"] == phylum_otu_count_df["phylum"][i]]
	tmp_phylum.set_index("Unnamed: 0", inplace = True, drop= True)
	del tmp_phylum["phylum"]
	colnames = tmp_phylum.columns.tolist()
	for col in colnames :
		if 'None' in col :
			del tmp_phylum[col]
	tmp_phylum = tmp_phylum.T
	tmp_phylum_corr = tmp_phylum.corr(method = 'spearman')
	tmp_phylum_otu_list = tmp_phylum_corr.index.tolist()
	tmp_phylum_otu_corr_list = []
	for otu in range(len(tmp_phylum_otu_list)) :
		tmp_phylum_otu_row_abs = tmp_phylum_corr.iloc[otu].abs()
		tmp_cumulative_corr = 1
		for j in range(len(tmp_phylum_otu_row_abs)) :
			tmp_cumulative_corr *= tmp_phylum_otu_row_abs[j]
		tmp_phylum_otu_corr_list.append(tmp_cumulative_corr ** (1/len(tmp_phylum_otu_row_abs)))
	tmp_phylum_otu_corr_df = pd.DataFrame({'otu' : tmp_phylum_otu_list, 'corr' : tmp_phylum_otu_corr_list})
	tmp_phylum_otu_corr_df = tmp_phylum_otu_corr_df.sort_values(by = 'corr', ascending = False)
	tmp_data_based_on_corr = pd.merge(tmp_phylum_otu_corr_df, data, left_on = "otu", right_on = "Unnamed: 0")
	del tmp_data_based_on_corr["corr"]
	del tmp_data_based_on_corr["Unnamed: 0"]
	del tmp_data_based_on_corr["phylum"]
	tmp_data_based_on_corr["cluster"] = i
	final_data_df = pd.concat([final_data_df, tmp_data_based_on_corr], axis = 0)	

final_data_df.to_csv(output_filename, mode = "w", index = False)

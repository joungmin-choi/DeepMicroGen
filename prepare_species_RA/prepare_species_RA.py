import pandas as pd 
import sys

method = sys.argv[1]

if method == "kraken" : 
	filelist = pd.read_csv('dataset_list.txt', header = None)
	for j in range(5) : #len(filelist)
		filename = filelist[0][j]
		data = pd.read_csv(filename, header = None, sep = '\t') 
		data = data[data[3] != 'R']
		data.reset_index(inplace = True, drop = True)
		abundance_list = []
		otu_name_list = []
		tmp_otu_name = ''
		for i in range(len(data)) : #len(data)
			if 'R' in data[3][i] :
				tmp_otu_name = data[5][i].strip() + '|'
			elif (data[3][i] == 'P') :
				tmp_otu_list = tmp_otu_name.split('|')
				tmp_otu_name = '|'.join(tmp_otu_list[:1]) + '|' + data[5][i].strip()
			elif (data[3][i] == 'C') : 
				tmp_otu_list = tmp_otu_name.split('|')
				tmp_otu_name = '|'.join(tmp_otu_list[:2]) + '|' + data[5][i].strip()
			elif (data[3][i] == 'O') : 
				tmp_otu_list = tmp_otu_name.split('|')
				tmp_otu_name = '|'.join(tmp_otu_list[:3]) + '|' + data[5][i].strip()
			elif (data[3][i] == 'F') :
				tmp_otu_list = tmp_otu_name.split('|')
				tmp_otu_name = '|'.join(tmp_otu_list[:4]) + '|' + data[5][i].strip()
			elif (data[3][i] == 'G') :
				tmp_otu_list = tmp_otu_name.split('|')
				tmp_otu_name = '|'.join(tmp_otu_list[:5]) + '|' + data[5][i].strip()
			elif (data[3][i] == 'S') : 
				tmp_otu_list = tmp_otu_name.split('|')
				tmp_otu_name = '|'.join(tmp_otu_list[:6]) + '|' + data[5][i].strip()
				abundance_list.append(data[0][i])
				otu_name_list.append(tmp_otu_name)
		preprocessed_output = pd.DataFrame({'percent' : abundance_list, 'otu' : otu_name_list})
		tmp_sum = preprocessed_output['percent'].sum()
		tmp_normalization_list = []
		for i in range(len(preprocessed_output)) :
			tmp_ra = (preprocessed_output['percent'][i]) / tmp_sum
			tmp_normalization_list.append(tmp_ra)
		preprocessed_output[filename] = tmp_normalization_list
		del preprocessed_output['percent']
		if j == 0:
			final_ouput = preprocessed_output.copy()
		else :
			final_ouput = pd.merge(final_ouput, preprocessed_output, left_on = 'otu', right_on = 'otu')

	final_ouput.to_csv('relative_abundance_data.csv', mode = 'w', index = False)

elif method == 'metaphlan' : 
	filename = sys.argv[2]
	data = pd.read_csv(filename, sep = '\t')
	clade_name = data['clade_name'].tolist()

	species_list = []
	for clade in clade_name : 
		if 's__' in clade :
			species_list.append(clade)

	species_df = pd.DataFrame({'species' : species_list})
	data_species = pd.merge(data, species_df, left_on = 'clade_name', right_on = 'species')
	del data_species['species']

	sample_list = data_species.columns.tolist()
	sample_list.remove('clade_name')
	sample_list.remove('NCBI_tax_id')

	for sample in sample_list :
		for i in range(len(data_species)) :
			data_species[sample][i] *= 0.01

	del data_species['NCBI_tax_id']
	data_species.to_csv('relative_abundance_data.csv', mode = 'w', index = False)



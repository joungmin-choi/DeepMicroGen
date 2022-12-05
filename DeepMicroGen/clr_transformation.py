import pandas as pd 
import numpy as np 
from skbio.stats. composition import clr
import sys

filename = sys.argv[1]
data = pd.read_csv(filename, index_col = 0)
data.reset_index(inplace = True, drop = False)

sample_list = data.columns.tolist()
sample_list.pop(0)

tmp_min = 100
for sample in sample_list :
	for i in range(len(data)) :
		if (tmp_min > data[sample][i]) and (data[sample][i] != 0.0) :
			tmp_min = data[sample][i] 
	pseudo_count = tmp_min/2 

for sample in sample_list :
	for i in range(len(data)) :
		if (data[sample][i] == 0.0) :
			data[sample][i] += pseudo_count

data.set_index(data.columns[0], inplace = True)
data = data.T 
clr_data = pd.DataFrame(clr(data))
clr_data.columns = data.columns 
clr_data.index = data.index
clr_data = clr_data.T
print(pseudo_count)

clr_data.to_csv("longitudinal_microbiome_data_clr.csv")





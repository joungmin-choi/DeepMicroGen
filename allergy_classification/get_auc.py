import pandas as pd 
from sklearn.metrics import roc_auc_score 
import numpy as np

allergy_list = ["milk", "egg", "peanut"]
data_type_list = ["_original", "_imputed"]

for data_type in data_type_list :
	result_df = pd.DataFrame()
	for allergy in allergy_list :
		result_list = []
		for i in range(1, 6) :
			label_file = "label_group_" + str(i) + "_" + allergy + data_type + ".csv"
			pred_file = "prediction_group_" + str(i) + "_" + allergy + data_type + ".csv"
			label = pd.read_csv(label_file, header = None)
			pred = pd.read_csv(pred_file, header = None)
			result_list.append(roc_auc_score(label, pred))
		result_df[allergy] = result_list
	result_df.to_csv("classification_auc_result" + data_type + ".csv", mode = "w", index = False)

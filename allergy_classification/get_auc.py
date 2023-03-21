import pandas as pd 
from sklearn.metrics import roc_auc_score 
import numpy as np

allergy_list = ["milk", "egg", "peanut"]
data_type_list = ["_original", "_imputed"]

for data_type in data_type_list :
	final_milk_df = pd.DataFrame()
	final_egg_df = pd.DataFrame()
	final_peanut_df = pd.DataFrame()
	for v in range(1,6) :
		dirname = "./v" + str(v) + "/"
		result_df = pd.DataFrame()
		for allergy in allergy_list :
			result_list = []
			for i in range(1, 6) :
				label_file = dirname + "label_group_" + str(i) + "_" + allergy + data_type + ".csv"
				pred_file = dirname + "prediction_group_" + str(i) + "_" + allergy + data_type + ".csv"
				label = pd.read_csv(label_file, header = None)
				pred = pd.read_csv(pred_file, header = None)
				result_list.append(roc_auc_score(label, pred))
			result_df[allergy] = result_list
		#result_df.to_csv(dirname + "classification_auc_result" + data_type + ".csv", mode = "w", index = False)
		final_milk_df["Experiment" + str(v)] = result_df["milk"]
		final_egg_df["Experiment" + str(v)] = result_df["egg"]
		final_peanut_df["Experiment" + str(v)] = result_df["peanut"]
	final_milk_df.loc['avg'] = final_milk_df.mean()
	final_egg_df.loc['avg'] = final_egg_df.mean()
	final_peanut_df.loc['avg'] = final_peanut_df.mean()
	final_milk_df.to_csv("results_milk" + data_type + ".csv", mode = "w", index = True)
	final_egg_df.to_csv("results_egg" + data_type + ".csv", mode = "w", index = True)
	final_peanut_df.to_csv("results_peanut" + data_type + ".csv", mode = "w", index = True)
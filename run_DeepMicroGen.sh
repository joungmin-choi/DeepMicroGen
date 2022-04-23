#!/bin/bash

input_profile="longitudinal_microbiome_data.csv"
input_mask="mask.csv"

learning_rate="0.001"
dropout_rate="0.7"
epochs="3000"


python3 convert_input.py $input_profile
python deepMicroGen.py "input_abundance_profiles.csv" $input_mask $learning_rate $dropout_rate $epochs


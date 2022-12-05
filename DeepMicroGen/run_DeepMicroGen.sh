#!/bin/bash

input_profile_RA="relative_abundance_data.csv"
input_profile_clr="longitudinal_microbiome_data_clr.csv"
input_mask="mask.csv"

learning_rate="0.001"
dropout_rate="0.7"
epochs="3000"

pseudo_count=$(python clr_transformation.py $input_profile_RA)
python convert_input.py $input_profile
python deepMicroGen.py "input_abundance_profiles.csv" $input_mask $learning_rate $dropout_rate $epochs $pseudo_count


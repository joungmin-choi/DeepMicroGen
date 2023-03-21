#!/bin/bash

###################################################
### Change "dir" as the dataset you want to run ###
###################################################

dir="./5cv_dataset/DeepMicroGen/"
#dir="./5cv_dataset/mean/"                          # mean imputation
#dir="./5cv_dataset/median/"                        # median imputation
#dir="./5cv_dataset/mice/"                          # MICE imputation
#dir="./5cv_dataset/moving_window/"                 # moving window-based imputation
#dir="./5cv_dataset/locf/"                          # Last observation carried forward imputation
#dir="./5cv_dataset/linear_curve_fitting/"          # Linear curve fitting
#dir="./5cv_dataset/cubic_curve_fitting/"           # Cubic curve fitting

for j in {1..5}
do
    mkdir "v"$j
    allergy_list="milk egg peanut"
    
    for allergy in $allergy_list
    do 
        for i in {1..5}
        do
        x_data=$dir"train_group_"$i"_X.csv"
        y_data=$dir"train_group_"$i"_"$allergy"_Y.csv"
        x_test=$dir"test_group_"$i"_X.csv"
        y_test=$dir"test_group_"$i"_"$allergy"_Y.csv"
        python allergy_classifier.py $allergy"_original" $i $x_data $y_data $x_test $y_test
        done
    done

    for allergy in $allergy_list
    do 
        for i in {1..5}
        do
        x_data=$dir"train_additional_group_"$i"_X.csv"
        y_data=$dir"train_additional_group_"$i"_"$allergy"_Y.csv"
        x_test=$dir"test_group_"$i"_X.csv"
        y_test=$dir"test_group_"$i"_"$allergy"_Y.csv"
        python allergy_classifier.py $allergy"_imputed" $i $x_data $y_data $x_test $y_test
        done
    done

    mv *.csv "v"$j
done

python get_auc.py
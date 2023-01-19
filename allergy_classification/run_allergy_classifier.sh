#!/bin/bash

dir="./5cv_dataset/"

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

python get_auc.py

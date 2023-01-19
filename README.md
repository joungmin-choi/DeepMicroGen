# DeepMicroGen
DeepMicroGen is a deep generative method for longitudinal microbiome data imputation. From the input dataset composed of multiple operational taxonomic units (OTUs), features incorporating the phylogenetic relationships between the taxonomies are extracted based on convolutional neural network (CNN) modules. These features are delivered to a bidirectional RNN based GAN model, and the imputed values are generated by learning the temporal dependency between the observations measured at different time points.

![Figure](https://github.com/joungmin-choi/DeepMicroGen/blob/main/fig_architecture.png?raw=true)

## Requirements
* Tensorflow (>= 1.8.0)
* Python (>= 2.7)
* Python packages : numpy, pandas, os, sys, scikit-learn

### Installation
To install the above requirments, please run below commands in the terminal :
```
pip install tensorflow==1.8.0 # For CPU
pip install tensorflow-gpu==1.8.0 # For GPU
pip install numpy pandas os sys
pip install -U scikit-learn
```

## Usage
Clone the repository or download source code files.

## Inputs
#### 1) Longituindal microbiome species-level relative abundance profiles
Prepare or edit **"relative_abundance_data.csv"** file having a matrix of species relative abundance profiles in **DeepMicroGen** directory, where each row and column represent OTU and sample ID, respectively. OTU should be in the format of **'kingdom|phylum|class|order|familiy|Genus|Species'** 
(e.g. *k__Bacteria|p__Actinobacteria|c__Actinobacteria|o__Corynebacteriales|f__Corynebacteriaceae|g__Corynebacterium|s__Corynebacterium_matruchotii*)

For the order of rows, the samples of each subject should be arranged in the increasing order of timepoints (tp) :

```
OTU,subject1_tp1,subject1_tp2,subject1_tp3,subject2_tp1...,subjectn_tp3
OTU1,0.00057,0.00034,0.00021,0.00012...,0.000023
OTU2,1.23337,1.00214,1.03028,2.02034...,1.003043
OTU3,0.00018,0.00024,0.02024,0.01011...,0.000029
...
```
For the outputs from Kraken2 and MetaPhlAn3, we provide a module to transform those to the species-level relative abundance profiles for DeepMicroGen. You can use those modules in **"prepare_species_RA"** directory.

* If you have the outputs from ***Kraken2***, please edit **"dataset_list.txt"** file having the list of the output filenames, and run the below command :
```
python3 prepare_species_RA.py kraken
```

* If you have the outputs from ***MetaPhlAn3***, please merge all the metphlan outpus in to one talbe, using **merge_metaphlan_tables.py** in MetaPhlAn package, and use that output. Replace "outputFileName" as your output file name and run the below command : 
```
python3 prepare_species_RA.py metaphlan outputFileName
```

#### 2) Mask matrix 
Prepare or edit **"mask.csv"** file in **DeepMicroGen** directory, indiciating whether each sampe is missing or not. Subject should be in a same order with the above profile dataset. The file should follow the below format, where the timepoints (tp1,tp2,tp3...) should be written in the integer or the float format :
```
SubjectName,tp1,tp2,tp3
Subject1,1,1,1
Subject2,1,0,1
Subject3,1,1,0
...
```

## Generating the imputed values for missing samples
Prepare the above two inputs in the **DeepMicroGen** directory and run the below command to generate the imputed values for missing samples. You can modify the learning rate, dropout rate, and epochs for training by modifying the options listed in **"run_DeepMicroGen.sh"** file.
```
./run_DeepMicroGen.sh
```

After training, you will get two final imputation outputs:  

**1) imputed_dataset_from_DeepMicroGen.csv** (clr-transformed format)  
**2) imputed_dataset_from_DeepMicroGen_scaled.csv** (relative abundance profiles where the sum of the abundance values in all OTUs equals to 1).

## Allergy classification ##
We uploaded the orginial clr-transformed species-level relative abundance profiles for DIABIMMUNE dataset and the imputation output from DeepMicroGen used for allergy prediction improvement in **"allergy_classification"** directory. For each subject, the profiles measured for 8 timepoints are aligned. The lstm-based allergy classifier used to predict the allergy in the expereiment was also uploaded. 

To re-run the allergy classification experiment in Section 3.6 of the manuscript, move to the **"allergy_classification"** directory and run the below command :
```
./run_allergy_classifier.sh
```

After running, you will get two result outputs : 
**1) average_classification_auc_result_original.csv**, and **2) average_classification_auc_result_imputed.csv**, which are average auc results for the allerge outcome predictions of the classifier trained with/without the addition of 25 imputed subjects, perforing 5-fold cross validation, respecitvely.

You can find the description for the other files below:
* diabimmune_allergy_metadata_original.csv, diabimmune_allergy_metadata_imputed.csv : Metadata (allergy information) for the subjects used for the classifier trained with/without the 25 imputed subjects, respectively.
* diabimmune_allergy_original_clr.csv, diabimmune_allergy_imputed_clr.csv : Clr-transformed dataset with/wihtout the addition of the imputed subjects
* 5cv_dataet : 5-fold cross validation dataset

## Contact
If you have any questions or problems, please contact to **joungmin AT vt.edu**.


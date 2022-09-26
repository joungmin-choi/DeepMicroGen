# DeepMicroGen
DeepMicroGen is a deep generative method for longitudinal microbiome data imputation. From the input dataset composed of multiple operational taxonomic units (OTUs), features incorporating the phylogenetic relationships between the taxonomies are extracted based on convolutional neural network (CNN) modules. These features are delivered to a bidirectional RNN based GAN model, and the imputed values are generated by learning the temporal dependency between the observations measured at different time points.

![Figure](https://github.com/joungmin-choi/DeepMicroGen/blob/main/fig_architecture.png?raw=true)

## Requirements
* Tensorflow (>= 1.8.0)
* Python (>= 2.7)
* Python packages : numpy, pandas, os, sys

### Installation
```
pip install tensorflow==1.8.0 # For CPU
pip install tensorflow-gpu==1.8.0 # For GPU
pip install numpy pandas os sys
```

## Usage
Clone the repository or download source code files.

## Inputs
### 1. Longituindal microbiome species-level relative abundance profiles
Prepare or edit **"longitudinal_microbiomd_data.csv"** file having a matrix of center-log transforemd species relative abundance profiles, where each row and column represent OTU and sample ID, respectively. For the order of rows, the samples of each subject should be arranged in the increasing order of timepoints (tp) :

```
OTU,subject1_tp1,subject1_tp2,subject1_tp3,subject2_tp1...,subjectn_tp3
OTU1,0.00057,0.00034,0.00021,0.00012...,0.000023
OTU2,1.23337,1.00214,1.03028,2.02034...,1.003043
OTU3,0.00018,0.00024,0.02024,0.01011...,0.000029
...
```

### 2. Mask matrix 
Prepare or edit **"mask.csv"** file indiciating whether each sampe is missing or not. Subject should be in a same order with the above profile dataset. The file should follow this format :
```
SubjectName,tp1,tp2,tp3
Subject1,1,1,1
Subject2,1,0,1
Subject3,1,1,0
...
```
## Generating the imputed values for missing samples
1. Run **'./run_DeepMicroGen.sh'** to generate the imputed values for missing samples. You can modify the learning rate, dropout rate, and epochs for training by modifying the options listed above in this file.
2. You can get the final output **'imputed_dataset_from_DeepMicroGen.csv'**.

## Contact
If you have any questions or problems, please contact to **joungmin AT vt.edu**.


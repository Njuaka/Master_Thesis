#!/bin/bash
#PBS -l walltime=10:00:00
#PBS -N eval_wbc_models
#PBS -m abe

#module load pandas
#module load scikit-learn
#module load matplotlib/3.2.1-foss-2020a
#module load Seaborn/0.10.1-foss-2020a
module load scikit-learn/1.1.2-foss-2022a
module load matplotlib/3.5.2-foss-2022a
module load Seaborn/0.12.1-foss-2022a

datapath="$VSC_DATA_VO/HDF5/WBC/"
runpath="$VSC_DATA_VO/Run/wbc/"
#FILENAME="$VSC_HOME/wbc_cellnames.txt"

python $VSC_HOME/final_thesis/evaluate_models3.py --run_dir $runpath --data_dir $datapath --type 'wbc' 

#python $VSC_HOME/final_thesis/evaluate_modelsplots.py --run_dir $runpath --data_dir $datapath --type 'wbc' 
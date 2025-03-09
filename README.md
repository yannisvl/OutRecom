# Experiments
Experiments to strengthen results for Neurips sumbission paper "Mechanism design augmented with output advice".
https://arxiv.org/abs/2406.14165

## Classes / Utils
This folder contains two types of files.
1. Algorithms: code for the optimal Facility Location algorithm and the Coordinatewise Median with Prediction mechanism
2. Datasets: classes that read input files for each dataset
3. Utils contains a file for handling 2D points

## Datasets
Datasets are not included in this folder because of the large space they require.
Please follow the following steps to download them and place them inside the "datasets" folder.
1. Download "2020_02_25.csv" from [https://www.kaggle.com/datasets/gidutz/autotel-shared-car-locations/data], extract the csv file and rename it to "autotel.csv"
2. Download "database.csv" file from [https://www.kaggle.com/datasets/usgs/earthquake-database], extract the csv file and rename it to "earthquake.csv"
3. Download 5 Twitter dataset files from [https://github.com/fe6Bc5R4JvLkFkSeExHM/k-center/tree/master/dataset], and use extract_twitter.py to extract it into a single csv
4. Download "loc-gowalla_totalCheckins.txt.gz" from [https://snap.stanford.edu/data/loc-Gowalla.html], extract the folder and place it inside the datasets directory
5. Download "loc-brightkite_totalCheckins.txt.gz" from [https://snap.stanford.edu/data/loc-Brightkite.html], extract the folder and place it inside the datasets directory

## Experiments
First run 
    `pip install -r requirements.txt`
to download required packages.

An experiment can be run with the "main.py" script with e.g. 
    `python main.py --problem "FL" --dataset "Autotel" --confidence 0.01`
Such a command runs Coordinatewise Median with Prediction with confidence 0.01 on the Autotel dataset with 100 different predictions. 
It also saves the results on a specific file inside the experiments folder.
Alternatively, the "run_exp.bat" script contains multiple experiments.

## Figures
The "process_data.py" script reads experimental results from the "experiments" folder and saves figures inside the "figures" folder.
Some of these are used in the original paper.

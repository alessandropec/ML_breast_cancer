# An exploration of ML methods for breast cancer classification
This repository contains the code for the exam project of "Mathematics for machine learning" course at Politecnico of Torino (years 2022).
Please refer to the uploaded Tesina (*tesina.pdf*) for more detail on the analysis and methods used.

## Dependencies
- **Python 3** (tested on python 3.7)
- [sklearn](https://scikit-learn.org)
- [numpy](https://numpy.org)
- [seaborn](https://seaborn.pydata.org)
- [pandas](https://pandas.pydata.org/)

## Dataset
The dataset used is uploaded in the repository and taken from https://archive.ics.uci.edu/ml/datasets/breast+cancer+wisconsin+(original)

## Script usage

Clone the repository, install dependencies, and run the following command for the full results.
```
python main.py --data=[path of dataset (./data/breast-cancer-wisconsin.data)]
```

### Run single phase of analysis and/or training
the script *main.py* can be used with the following arguments:
```
  -h, --help            
                        show this help message and exit
                        
  --data REQUIRED
                        The path of the file containing the dataset
                        
  --run_analysis DEFAULT: [all]
                        The analysis to run:[no]: skip the analysis [ext]:
                        scatter of 2 extracted features [pca]: run only pca
                        analysis [correlation]: run only correlation analysis
                        [all]: run all analysis
                        
  --run_training DEFAULT: [all]
                        The model to train:[no]: skip the training [rf]: random
                        forest [svc]: support vector machine [lr]: logistic
                        regression [knn]: k nearest neighboors [all]: all
                        models
                        
  --test_size DEFAULT: [0.33]
                        The percentage of the test set
  ```

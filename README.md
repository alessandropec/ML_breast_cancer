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

## Model results

The best result is achieved by the Random forest with the extracted features, this give an accuracy of 96%


|                    | SVC                                                            | SVC (ext. features)                                           | Random Forest                                                                                                                      | **Random Forest (ext. features)**                                                                                                  | Log Regression                                                              | Log Regression (ext. features)                                              | KNN                                                 | KNN (ext. features) |
|--------------------|----------------------------------------------------------------|---------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------|-----------------------------------------------------------------------------|-----------------------------------------------------|---------------------|
| Accuracy           | 0,94                                                           | 0,95                                                          | 0,94                                                                                                                               | **0,96**                                                                                                                           | 0,95                                                                        | 0,94                                                                        | 0,93                                                | 0,94                |
| Precision (benign) | 0,95                                                           | 0,95                                                          | 0,94                                                                                                                               | 0,97                                                                                                                           | 0,95                                                                        | 0,94                                                                        | 0,92                                                | 0,94                |
| Precision (malign) | 0,93                                                           | 0,94                                                          | 0,94                                                                                                                               | 0,95                                                                                                                           | 0,94                                                                        | 0,94                                                                        | 0,94                                                | 0,94                |
| Recall (benign)    | 0,94                                                           | 0,95                                                          | 0,95                                                                                                                               | 0,95                                                                                                                           | 0,95                                                                        | 0,95                                                                        | 0,95                                                | 0,95                |
| Recall (malign)    | 0,94                                                           | 0,94                                                          | 0,93                                                                                                                               | 0,97                                                                                                                           | 0,94                                                                        | 0,93                                                                        | 0,9                                                 | 0,93                |
| F1 (benign)        | 0,94                                                           | 0,95                                                          | 0,95                                                                                                                               | 0,96                                                                                                                           | 0,95                                                                        | 0,95                                                                        | 0,93                                                | 0,95                |
| F1 (malign)        | 0,94                                                           | 0,94                                                          | 0,94                                                                                                                               | 0,96                                                                                                                           | 0,94                                                                        | 0,94                                                                        | 0,92                                                | 0,94                |
| support (benign)   | 82                                                             | 82                                                            | 82                                                                                                                                 | 82                                                                                                                             | 82                                                                          | 82                                                                          | 82                                                  | 82                  |
| support (malign)   | 71                                                             | 71                                                            | 71                                                                                                                                 | 71                                                                                                                             | 71                                                                          | 71                                                                          | 71                                                  | 71                  |
| Best param         | {'svc__C': 10, 'svc__gamma': 'scale', 'svc__kernel': 'linear'} | {'svc__C': 0.5, 'svc__gamma': 'scale', 'svc__kernel': 'poly'} |  {'rf__criterion': 'entropy', 'rf__max_depth': 5, 'rf__max_features': 'sqrt', 'rf__min_samples_split': 10, 'rf__n_estimators': 50} |  {'rf__criterion': 'gini', 'rf__max_depth': 8, 'rf__max_features': 'sqrt', 'rf__min_samples_split': 2, 'rf__n_estimators': 50} | {'lr__C': 1, 'lr__dual': False, 'lr__max_iter': 500, 'lr__penalty': 'none'} | {'lr__C': 1, 'lr__dual': False, 'lr__max_iter': 500, 'lr__penalty': 'none'} | {'knn__n_neighbors': 3, 'knn__weights': 'uniform'}  | {'knn__n_neighbors': 3, 'knn__weights': 'distance'}                    |

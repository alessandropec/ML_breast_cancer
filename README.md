# An exploration of ML methods for breast cancer classification
This repository contains the code for the exam project of "Mathematics for machine learning" course at Politecnico of Torino (years 2022)

##Dependencies

## Script usage

Clone the repository install dependencies and run the following command for the full results.
```
python main.py --data_dir=[path of dataset (./data/breast-cancer-wisconsin.data)]
```

### Run single phase of analysis and/or training
the script *main.py* can be used with the following arguments:
```
  -h, --help            
                        show this help message and exit
                        
  --data_dir REQUIRED
                        The path of .txt containing the dataset
                        
  --run_analysis DEFAULT: [all]
                        The analysis to run:[no]: skip the analysis [ext]:
                        scatter of 2 extracted features [pca]: run only pca
                        analysis [correlation]: run only correlation analysis
                        [all]: run all analysis
                        
  --run_training DEFAULT: [all]
                        The model to train:[no]: skip the training[rf]: random
                        forest [svc]: support vector machine [lr]: logistic
                        regression [knn]: k nearest neighboors [all]: all
                        models
                        
  --test_size DEFAULT: [0.33]
                        The percentage of the test set
  ```

from email.policy import default
import pandas as pd
import matplotlib.pyplot as plt
from itertools import combinations
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import numpy as np

import argparse

from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier as KNN

from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin

from sklearn.metrics import classification_report


def load_dataset(path):
    '''**********************************Load dataset************************************'''
    #class 2: benign 4:malign
    df=pd.read_csv(path,\
                    header=None,\
                    delimiter=",",\
                    names=["patience_id","clump_thickness", "uniformity_of_cell_size", \
                            "uniformity_of_cell_shape","marginal_adhesion",\
                            "single_epithelial_cell_size","bare_nuclei","bland_chromatin",\
                            "normal_nucleoli","mitoses","class"],\
                   index_col=False)

    return df
def correlation_analysis(df,row,col,add_title=""):
    '''**********************************Correlation matrix and scatters plotting************************************'''
    plt.rc('xtick', labelsize=8) 
    plt.rc('ytick', labelsize=8) 


   
    #Correlation matrix
    fig1=plt.figure(figsize=(9, 11))
    
    corr=df.corr()
    
    heatmap=sns.heatmap(corr,annot=True,vmin=-1,vmax=1,square=True,cmap="bwr")
    fig1.tight_layout()
    heatmap.set_title('Correlation Heatmap '+add_title, fontdict={'fontsize':12}, pad=20)
    #heatmap.set_xticklabels(heatmap.get_xticklabels(),rotation = 10)
    #Correlation scatters of all combinations
    df_no_c=df.drop(columns=["class"])
    features=df_no_c.columns
    cc=list(combinations(features,2))
    print(f"\nNumber of group of 2 {add_title}: ",len(cc),"\nList:\n",cc)
    
  
    #TO DO: set legend of colors, fix the text in order to no overlap, order the subplots like the correlation matrix
    fig, axes = plt.subplots(row,col, figsize=(20,20),sharex=True, sharey=True) #6*6=36 total number of group of 2 features

    fig.suptitle('Correlation analysis of all combinations of 2 features '+add_title, fontsize=16)
    axes=axes.ravel()
    
    for i in range(0, len(cc)):
        axes[i].set_ylabel(cc[i][1],fontsize = 5) # Y label
        axes[i].set_xlabel(cc[i][0],fontsize = 5) # X label
        sns.regplot(x=df_no_c[cc[i][0]],y=df_no_c[cc[i][1]],scatter_kws={'alpha':0.2,"color":tuple(df["class"].apply(lambda x: "green" if x=="b" else "red").values)},ax=axes[i])
    
    line1 = plt.Line2D(range(1), range(1), color="white", marker='o', markerfacecolor="green")
    line2 = plt.Line2D(range(1), range(1), color="white", marker='o',markerfacecolor="red")
    axes[0].legend(handles=(line1,line2),labels=["Benign","Malign"],loc=(-1,+1.2))  



def mypca(df,c=1,add_title=""):
    pca=PCA(n_components=c)
    #df_tmp=df.drop(columns=["class"])

    cols=[]
    for i in range(c):
        cols.append(f"D{i}")
 
    redu=pca.fit(df).transform(df)
    redu_df=pd.DataFrame(redu,columns=cols)
    #redu_df=pd.concat([redu_df,df["class"]],axis=1)

      # Calculate the variance explained by priciple components
    print("Variance explained "+add_title)
    print('Variance of each component:', pca.explained_variance_ratio_)
    print('Total Variance Explained:', round(sum(list(pca.explained_variance_ratio_))*100, 2))
    print()

    load=[]
    for i in range(c):
        load.append(redu_df[f'D{i}'])
    return load,pca

def pca_analysis(df,target,add_title=""):

    (x,y,z),pca=mypca(df,c=3,add_title=add_title)
    
    sns.set(style = "darkgrid")

    fig = plt.figure()
    fig.suptitle("PCA analysis in 1d, 2d, and 3d spaces "+add_title)
    ax1=fig.add_subplot(131)
    ax1.set_xlabel("1st Principal comp.")
    ax1.set_ylabel("Jitter")
    y_j = 0.1 * np.random.rand(len(x)) -0.05 #jitter
    ax1.scatter(x, y_j,c=[ "green" if v=="b" else "red" for v in target],alpha=0.3)
    
    line1 = plt.Line2D(range(1), range(1), color="white", marker='o', markerfacecolor="green")
    line2 = plt.Line2D(range(1), range(1), color="white", marker='o',markerfacecolor="red")
    ax1.legend(handles=(line1,line2),labels=["Benign","Malign"],loc=(-0.5,1))  

    ax2 = fig.add_subplot(132)
    ax2.set_xlabel("1st Principal comp.")
    ax2.set_ylabel("2nd Principal comp.")
    

    ax2.scatter(x, y,c=[ "green" if v=="b" else "red" for v in target ],alpha=0.3)
    
    ax3 = fig.add_subplot(133,projection = '3d')
    ax3.set_xlabel("1st Principal comp.")
    ax3.set_ylabel("2nd Principal comp.")
    ax3.set_zlabel("3rd Principal comp.")

    ax3.scatter(x, y,z,c=[ "green" if v=="b" else "red" for v in target],alpha=0.3)



    return (x,y,z),pca
    


def prepare_training(df,test_size):
    X_train, X_test, y_train, y_test = train_test_split(df.drop(columns=["class"]), df["class"], test_size=test_size, random_state=42)
    print("\nSpillting dataset with test size:",test_size,"\nLenght of train:",len(X_train),"\nLenght of test:",len(X_test))
    return X_train, X_test, y_train, y_test 

def train_test_model(pipe,params,df, X_test, y_test,add_title=""):
    

    gs=GridSearchCV(pipe, params)

    est=gs.fit(df.drop(columns=["class"]),df["class"])
    cross_score=est.best_score_
    print("\nCross validation results:\nPipeline:\n",est,"\nBest param:\n",est.best_params_,"\nF1 avg cross val score: ",cross_score)

    pred=est.predict(X_test)
   
    creport=classification_report(y_test,pred)
    print("\nClassification report for test:\n",creport)
    conf_matrix_test=confusion_matrix(y_test,pred)
   
    fig2=plt.figure()
    fig2.suptitle("Confusion matrix "+add_title)
    ax=sns.heatmap(conf_matrix_test,annot=True,cmap="Blues",yticklabels=["b (negative)","m (positive)"],xticklabels=["b (negative)","m (positive)"])
    ax.set_xlabel("Actual values")
    ax.set_ylabel("Predicted values")

    return  cross_score,creport,conf_matrix_test,est.best_estimator_

class DistExtractor(BaseEstimator,TransformerMixin):
    b_c=0
    m_c=0
    
    def __init__(self) -> None:
        super().__init__()
    
    def fit(self,X,y=None):
        #print(y)
        new_df=pd.DataFrame(X.copy())
        y=pd.DataFrame(y.copy())
        y.index=range(len(new_df))
        #print(y)
        benign=new_df.loc[y["class"]=="b"]
        malign=new_df.loc[y["class"]=="m"]

        self.b_c=list(benign.mean().values)
        self.m_c=list(malign.mean().values)

        return self

    def transform(self,X,y=None):
        new_df=pd.DataFrame(X)
        #print(new_df)
        new_df["b_dist"]= (new_df - np.array(self.b_c)).pow(2).sum(1).pow(0.5)

        new_df["m_dist"]= (new_df.drop(columns=["b_dist"]) - np.array(self.m_c)).pow(2).sum(1).pow(0.5)
        new_df.columns=["clump_thickness", "uniformity_of_cell_size", \
                            "uniformity_of_cell_shape","marginal_adhesion",\
                            "single_epithelial_cell_size","bare_nuclei","bland_chromatin",\
                            "normal_nucleoli","mitoses","b_dist","m_dist"]
        return new_df

def basic_clean(df):
    
    df=df.drop(columns=["patience_id"]) #Patience id is useless
   
    #drop duplicates
    print("\nDropping duplicates:")
    print("Len with duplicates:",len(df))
    df=df.drop_duplicates(keep="first")
    print("Len without duplicates:",len(df),"\n")
    df.index=range(len(df)) #adjust indexing after drop
    #change class to categorical attribute
    df["class"]=df["class"].apply(lambda x: "b" if x==2 else "m").astype("category")
    print("Check for balanced dataset:")
    print(df["class"].value_counts(),"\n")
    df=df.replace("?",np.NaN)
   
    print("Presence of null value (\"NaN\" in each features):\n",df.isin([np.NaN]).any(),"\nNOTE: The NaN value will be replaced with median strategy.")
    
    print("\nDataset:\n",df)
    return df

def data_exploration(X_train,y_train,imp,ext,cmd):
    #use the first two estimator of pipeline to process data and plot analysis
    imp_step=imp.fit_transform(X_train)
    imp_dist_step=ext.fit_transform(imp_step,y_train)
    if cmd=="pca" or cmd=="all":
        #PCA analysis of 1st 2nd and 3rd Principal components
        print("\n***********PCA ANALYSIS****************\n")    
        pca_analysis(imp_step,y_train, add_title="without extracted features") #without extracted features
        pca_analysis(imp_dist_step,y_train,add_title="with extracted features") #with extracted features

    if cmd=="ext" or cmd=="all":
        print("\n***********EXTRACTED FEATURES ANALYSIS****************\n")
        #Scatter of 2 new extracted features
        print(imp_dist_step)
        fig=plt.figure()
        fig.suptitle("Scatter of 2 new extracted features (only training)")
        ax=fig.add_subplot(111)
        ax.scatter(imp_dist_step["b_dist"],imp_dist_step["m_dist"],c=["red" if s=="m" else "green" for s in y_train],alpha=0.5)
        ax.set_xlabel("Distance from benign centroid")
        ax.set_ylabel("Distance from malign centroid")
        line1 = plt.Line2D(range(1), range(1), color="white", marker='o', markerfacecolor="green")
        line2 = plt.Line2D(range(1), range(1), color="white", marker='o',markerfacecolor="red")
        ax.legend(handles=(line1,line2),labels=["Benign","Malign"])

    if cmd=="correlation" or cmd=="all":
        print("***********CORRELATION ANALYSIS****************\n")
        #Correlation analysis
        y_train.index=range(len(y_train))
        df_train_no_ext=pd.concat([pd.DataFrame(imp_step,columns=X_train.columns),y_train],axis=1)
        
        correlation_analysis(df_train_no_ext,6,6,add_title="without extracted features")

        df_train_ext=pd.concat([imp_dist_step,y_train],axis=1)
        correlation_analysis(df_train_ext,5,11,add_title="with extracted features")

def training(pipeIDM,pipeIM,df_train,X_test,y_test,cmd):
    if cmd=="rf" or cmd=="all":
        #Random Forest
        print("**************Random Forest*************************")
        rf=RandomForestClassifier(random_state=42)
        tmpIDM=list(pipeIDM.steps) #Copy the full ipeline
        tmpIDM.append(("rf",rf)) #Append the model
        tmpIM=list(pipeIM.steps) #Same for the pipeline without feature extraction
        tmpIM.append(("rf",rf))
        pipeIDRF=Pipeline(tmpIDM) #Instntiate pipeline object
        pipeIRF=Pipeline(tmpIM)
        parameters = {'rf__max_depth':(1,2,5,8,None),
                    'rf__n_estimators':(1,5,10,50,100,500),
                    'rf__min_samples_split':(2,5,10),
                    "rf__max_features": [None, "sqrt"],
                    "rf__criterion": ["gini", "entropy"]}
        #Train and test in cross validation grid search and test with test data
        _,_,_,est=train_test_model(pipeIDRF,parameters,df_train,X_test,y_test,add_title="Random Forest with extracted features")#With new features
        fi=pd.DataFrame(est.steps[-1][1].feature_importances_,index=cols,columns=["Importance"])
        print(f"Feature importance:\n {fi.head(11)}")

        print("\n******Without extracted features********\n")
        _,_,_,est=train_test_model(pipeIRF,parameters,df_train,X_test,y_test,add_title="Random Forest without extracted features")#Without new features
        fi=pd.DataFrame(est.steps[-1][1].feature_importances_,index=X_train.columns,columns=["Importance"])
        print(f"Feature importance:\n {fi.head(9)}")

    if cmd=="svc" or cmd=="all":
        #SVC
        print("**************Support Vector Classifier*************************")
        svc=SVC(random_state=42)
        tmpIDM=list(pipeIDM.steps) #Copy the full ipeline
        tmpIDM.append(("svc",svc)) #Append the model
        tmpIM=list(pipeIM.steps) #Same for the pipeline without feature extraction
        tmpIM.append(("svc",svc))
        pipeIDSVC=Pipeline(tmpIDM) #Instntiate pipeline object
        pipeISVC=Pipeline(tmpIM)
        parameters = {'svc__kernel':('linear', 'poly', 'rbf', 'sigmoid'),
                    'svc__C':(0.1,0.5,1.0,10,50),
                    'svc__gamma':('scale','auto')
                    }
        train_test_model(pipeIDSVC,parameters,df_train,X_test,y_test,add_title="SVC with extracted features")
        print("\n******Without extracted features********\n")
        train_test_model(pipeISVC,parameters,df_train,X_test,y_test,add_title="SVC without extracted features")
    
    if cmd=="lr" or cmd=="all":
        #Logistic Regression TO DO: il solver rbfg supporta solo l2 o None sistemare i grid parameter
        print("**************Logistic Regression*************************")
        lr=LogisticRegression(random_state=42)
        tmpIDM=list(pipeIDM.steps) #Copy the full ipeline
        tmpIDM.append(("lr",lr)) #Append the model
        tmpIM=list(pipeIM.steps) #Same for the pipeline without feature extraction
        tmpIM.append(("lr",lr))
        pipeIDLR=Pipeline(tmpIDM) #Instntiate pipeline object
        pipeILR=Pipeline(tmpIM)
        parameters = {'lr__penalty':('none', 'l2'),
                    'lr__dual':([False]),
                    "lr__C": [1, 10, 100],
                    'lr__max_iter':[500,1000]

                    }
        train_test_model(pipeIDLR,parameters,df_train,X_test,y_test,add_title="Log Regression with extracted features")
        print("\n******Without extracted features********\n")
        train_test_model(pipeILR,parameters,df_train,X_test,y_test,add_title="Log Regression without extracted features")

    if cmd=="knn" or cmd=="all":
        #K nearest neighboor
        print("**************KNN*************************")
        knn=KNN()
        tmpIDM=list(pipeIDM.steps) #Copy the full ipeline
        tmpIDM.append(("knn",knn)) #Append the model
        tmpIM=list(pipeIM.steps) #Same for the pipeline without feature extraction
        tmpIM.append(("knn",knn))
        pipeIDKNN=Pipeline(tmpIDM) #Instntiate pipeline object
        pipeIKNN=Pipeline(tmpIM)
        parameters = {'knn__n_neighbors':(1,2, 3,5,10),
                    'knn__weights':("uniform","distance"),
                    }
        train_test_model(pipeIDKNN,parameters,df_train,X_test,y_test,add_title="K nearest neighboors with extracted features")
        print("\n******Without extracted features********\n")
        train_test_model(pipeIKNN,parameters,df_train,X_test,y_test,add_title="K nearest neighboors without extracted features")



   

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description='Analysis/Exploration of the dataset and training of the main ML model '
    )
    parser.add_argument('--data_dir', required=True, help='The path of .txt containing the dataset')
     parser.add_argument('--run_analysis',required=True,default='all',help="The analysis to run:[no]: skip the analysis [ext]: scatter of 2 extracted features [pca]: run only pca analysis [correlation]: run only correlation analysis [all]: run all analysis")
    parser.add_argument('--run_training',required=True,default='all',help="The model to train:[no]: skip the training[rf]: random forest [svc]: support vector machine [lr]: logistic regression [knn]: k nearest neighboors [all]: all models")
    parser.add_argument('--test_size',default=0.33,help="The percentage of the test set")
    args = parser.parse_args()
    print("*********LOADING, BASIC CLEANING & SLITTING***************")
    #LOADING & BASIC CLEANING
    df=load_dataset(args.data_dir)#Load dataset
    df=basic_clean(df)#Drop duplicates check for null and change null ('?') to np.NaN
    
    #DATASET SPLITTING (TRAIN-TEST)
    X_train,X_test,y_train,y_test=prepare_training(df,test_size=args.test_size)#Split into train and test
    df_train=pd.concat([X_train,y_train],axis=1)#Instantiate dataframe of training data
    
   

    #Prepro pipeline
    imp=SimpleImputer(missing_values=np.NaN,strategy="median")#Replace NaN with median
    ext=DistExtractor()#Extract centroids and add 2 feature distance for beningn centroids and malign centroids
    pipeIDM=Pipeline(steps=[("Imputer",imp),("Distance extractor",ext)])#with new feature extracted
    pipeIM=Pipeline(steps=[("Imputer",imp)])#Without feature extracted

    cols=pipeIDM.fit_transform(X_train,y_train).columns#get the columns after preprocessing (to printo below)

    #Data analysis and exploration
    if args.run_analysis!='no':
        data_exploration(X_train,y_train,imp,ext,args.run_analysis)
        plt.show()

    if args.run_training!='no':
        training(pipeIDM,pipeIM,df_train,X_test,y_test,args.run_training)
        plt.show()

    exit()
  

    
    

   
   
 


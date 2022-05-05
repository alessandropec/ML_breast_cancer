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
from sklearn.metrics import classification_report

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin

from sklearn.neural_network import MLPClassifier


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
def correlation_analysis(df,add_title=""):
    '''**********************************Correlation matrix and scatters plotting************************************'''
    plt.rc('xtick', labelsize=8) 
    plt.rc('ytick', labelsize=8) 


   
    #Correlation matrix
    fig1=plt.figure()
    
    corr=df.corr()
    
    heatmap=sns.heatmap(corr,annot=True,vmin=-1,vmax=1,square=True,cmap="bwr")
    fig1.tight_layout()
    fig1.tight_layout()
    heatmap.set_title('Correlation Heatmap '+add_title, fontdict={'fontsize':12}, pad=20)
    #heatmap.set_xticklabels(heatmap.get_xticklabels(),rotation = 10)
    #Correlation scatters of all combinations
    df_no_c=df.drop(columns=["class"])
    features=df_no_c.columns
    cc=list(combinations(features,2))
    #print(f"\nNumber of group of 2 features {add_title}: ",len(cc))
    #print("List:\n",cc)

    sns.set(font_scale = 0.7)

    g=sns.pairplot(df.iloc[: ,:], hue = 'class',palette={"b":[0,1,0,0.5],"m":[1,0,0,0.5]},corner=True)
    
    g.tight_layout()
    g.tight_layout()

    for i,axes in enumerate(g.axes.flat):
       
        if axes==None: #To fix with corener=True
            continue
        axes.set_ylabel(axes.get_ylabel(), rotation=0, horizontalalignment='right')
        axes.set_xlabel(axes.get_xlabel(), rotation=45, horizontalalignment='right')
    '''
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
    '''


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
    print('Explained variance of each component:', pca.explained_variance_ratio_)
    print('Total Variance Explained:', round(sum(list(pca.explained_variance_ratio_))*100, 2))

    print(f"Variance of each component:\n{redu_df[0:3].var()}")
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

    fig = plt.figure()
    fig.suptitle("PCA analysis in 1d, 2d, and 3d spaces "+add_title)
    ax2=fig.add_subplot(111)
    
    line1 = plt.Line2D(range(1), range(1), color="white", marker='o', markerfacecolor="green")
    line2 = plt.Line2D(range(1), range(1), color="white", marker='o',markerfacecolor="red")
    ax2.legend(handles=(line1,line2),labels=["Benign","Malign"],loc=(-0.5,1))  

    
    ax2.set_xlabel("1st Principal comp.")
    ax2.set_ylabel("2nd Principal comp.")
    

    ax2.scatter(x, y,c=[ "green" if v=="b" else "red" for v in target ],alpha=0.3)
    
 



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

    pred_train=est.predict(df.drop(columns=["class"]))
    #log=est.predict_proba(df.drop(columns=["class"]))
  
   
    print(est.classes_)
    
   
    creport=classification_report(df["class"],pred_train)
    print("\nClassification report for TRAIN:\n",creport)


    pred=est.predict(X_test)
   
    creport=classification_report(y_test,pred)
    print("\nClassification report for TEST:\n",creport)
    conf_matrix_test=confusion_matrix(y_test,pred)
   
    fig2=plt.figure()
    fig2.suptitle("Confusion matrix "+add_title)
    ax=sns.heatmap(conf_matrix_test,annot=True,cmap="Blues",yticklabels=["b (negative)","m (positive)"],xticklabels=["b (negative)","m (positive)"])
    ax.set_xlabel("Actual values")
    ax.set_ylabel("Predicted values")

    return  cross_score,creport,conf_matrix_test,est.best_estimator_

class DfBuilder(BaseEstimator,TransformerMixin):
    
    def __init__(self) -> None:
        super().__init__()
       
    
    def fit(self,X,y=None):
        
        return self
    def transform(self,X,y=None):

        
        df=pd.DataFrame(X,columns=["clump_thickness", "uniformity_of_cell_size", \
                            "uniformity_of_cell_shape","marginal_adhesion",\
                            "single_epithelial_cell_size","bare_nuclei","bland_chromatin",\
                            "normal_nucleoli","mitoses","b_dist","m_dist"])
        return df
    

class DistExtractor(BaseEstimator,TransformerMixin):
    b_c=0
    m_c=0
    onlyCD=None
    
    def __init__(self,onlyCD=False) -> None:
        self.onlyCD=onlyCD
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
        if self.onlyCD:
            return new_df[["b_dist","m_dist"]]
        else: return new_df

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
    print("Number of NaN for bare nuclei attribute:",df['bare_nuclei'].isna().sum())
    print("\nDataset:\n",df)
    return df

def data_exploration(X_train,y_train,imp,extAll,scaler,cmd):
    #use the first two estimator of pipeline to process data and plot analysis
    imp_step=scaler.fit_transform(imp.fit_transform(X_train))
    #imp_dist_step=pd.DataFrame(scaler.fit_transform(ext.fit_transform(imp_step,y_train)),columns=cols)
    imp_dist_step=extAll.fit_transform(imp_step,y_train)
    print(imp_dist_step)
    imp_dist_scale_step=scaler.fit_transform(imp_dist_step)
    print(imp_dist_scale_step)

    #imp_dist_scale_step=pd.DataFrame(scaler.fit_transform(imp_dist_step),columns=imp_dist_step.columns)
  
    if cmd=="pca" or cmd=="all":
        #PCA analysis of 1st 2nd and 3rd Principal components
        print("\n***********PCA ANALYSIS****************\n")    
        pca_analysis(imp_step,y_train, add_title="without extracted features") #without extracted features
        pca_analysis(imp_dist_scale_step,y_train,add_title="with extracted features") #with extracted features

    if cmd=="ext" or cmd=="all":
        print("\n***********EXTRACTED FEATURES ANALYSIS****************\n")
        #Scatter of 2 new extracted features
        print(imp_dist_step.describe())
       
        #print(imp_dist_scale_step)
        fig=plt.figure()
        fig.suptitle("Scatter of 2 new extracted features (only training)")
        ax=fig.add_subplot(111)
        ax.scatter(imp_dist_step["b_dist"],imp_dist_step["m_dist"],c=["red" if s=="m" else "green" for s in y_train],alpha=0.5)
        ax.set_xlabel("Distance from benign centroid")
        ax.set_ylabel("Distance from malign centroid")
        line1 = plt.Line2D(range(1), range(1), color="white", marker='o', markerfacecolor="green")
        line2 = plt.Line2D(range(1), range(1), color="white", marker='o',markerfacecolor="red")
        ax.legend(handles=(line1,line2),labels=["Benign","Malign"])
        print("Variance of two new features:\n")
        print(f"Variance of each component:\n{imp_dist_step[['b_dist','m_dist']].var()}")
       
     

    if cmd=="correlation" or cmd=="all":
        print("***********CORRELATION ANALYSIS****************\n")
        #Correlation analysis
        y_train.index=range(len(y_train))
        #df_train_no_ext=pd.concat([pd.DataFrame(imp_step,columns=X_train.columns),y_train],axis=1)
        
        #correlation_analysis(df_train_no_ext,6,6,add_title="without extracted features")

        df_train_ext=pd.concat([imp_dist_step,y_train],axis=1)
        correlation_analysis(df_train_ext,add_title="with extracted features")

def scatter_hard_sample(model,df_train,pipeIDM,prob=False,add_txt=""):
    
    df_x=df_train.drop(columns="class")
    pro_df=pipeIDM.fit_transform(df_x,df_train["class"])
   
    pred=model.predict(df_x)
    
 

    #print(pred,pro_df,df_train)
    tmp=df_train["class"].copy()
    tmp.index=range(len(df_train))
 
    mask= pred==tmp
   
  
    pro_df["right"]=mask
    
    pro_df["class"]=tmp
   
    pro_df["class"]=pro_df["class"].cat.add_categories(["false benign","false malign"])
    pro_df.loc[(pro_df["right"]==False) & (pro_df["class"]=="b"),"class"]="false malign"
    pro_df.loc[(pro_df["right"]==False) & (pro_df["class"]=="m"),"class"]="false benign"

    fig=plt.figure()
    fig.suptitle("Scatterplot of extracted features with missclassified samples "+add_txt)

   
    sns.scatterplot(data=pro_df, x="b_dist", y="m_dist",hue='class',\
        palette={"b":[0,1,0,0.25],"m":[1,0,0,0.25],"false benign":[0,1,0,1],"false malign":[1,0,0,1]},\
        style="class",\
        markers={"false benign":"X","false malign":"X","b":"o","m":"o"})

    if prob:
        log=model.predict_proba(df_x)
        prob_mask=prob_mask=[True if np.abs(arr[0]-arr[1])<=0.8 else False for arr in log ]
        log=log[prob_mask]
        def label_point(x, y, val, ax):
        
            a = pd.concat({'x': x, 'y': y,}, axis=1)
            a.index=range(len(a))
            for i, point in a.iterrows():
              
                ax.text(point['x']+.02, point['y']+0.02, str(f"b:{val[i,0]:.{2}f} m:{val[i,1]:.{2}f}"),fontsize=6)


        label_point(pro_df.loc[prob_mask]["b_dist"], pro_df.loc[prob_mask]["m_dist"], log, plt.gca()) 

  
    plt.legend(title="Legend",fontsize=8)
    #plt.scatter(x=hard_sample["b_dist"],y=hard_sample["m_dist"],c=hard_sample["actual"])


def training(pipeIDM,pipeIM,pipeIDMS,df_train,X_test,y_test,cmd):
    if cmd=="rf" or cmd=="all":
        #Random Forest
        print("**************Random Forest*************************")
        rf=RandomForestClassifier(random_state=42)
        tmpIDMS=list(pipeIDMS.steps) #Copy the full ipeline
        tmpIDMS.append(("rf",rf)) #Append the model

        tmpIDM=list(pipeIDM.steps) #Copy the full ipeline
        tmpIDM.append(("rf",rf)) #Append the model


        tmpIM=list(pipeIM.steps) #Same for the pipeline without feature extraction
        tmpIM.append(("rf",rf))

        pipeIDRFS=Pipeline(tmpIDMS) #Instntiate pipeline object
        pipeIDRF=Pipeline(tmpIDM)
        pipeIRF=Pipeline(tmpIM)
        #parameters = {'rf__criterion': ['gini'], 'rf__max_depth': [8], 'rf__max_features': ['sqrt'], 'rf__min_samples_split': [2], 'rf__n_estimators': [50]}
        
        parameters={'rf__max_depth':(1,2,5,8,None),
                    'rf__n_estimators':(1,5,10,50,100,500),
                    'rf__min_samples_split':(2,5,10),
                    "rf__max_features": [None, "sqrt"],
                    "rf__criterion": ["gini", "entropy"]}
        
        #Train and test in cross validation grid search and test with test data
        print("\n******With extracted features and original********\n")
        _,_,_,bestEst=train_test_model(pipeIDRFS,parameters,df_train,X_test,y_test,add_title="Random Forest with extracted and original features")#With new features
        scatter_hard_sample(bestEst,df_train,pipeIDMS,prob=True,add_txt="CD and original features (train)")
        scatter_hard_sample(bestEst,pd.concat([X_test,y_test],axis=1),pipeIDMS,prob=True,add_txt="CD and original features (test)")
        fi=pd.DataFrame(bestEst.steps[-1][1].feature_importances_,index=cols,columns=["Importance"])
        print(f"Feature importance:\n {fi.head(11)}")

        print("\n******With only extracted features********\n")
        _,_,_,bestEst=train_test_model(pipeIDRF,parameters,df_train,X_test,y_test,add_title="Random Forest with extracted features")#With new features
        scatter_hard_sample(bestEst,df_train,pipeIDM,prob=True,add_txt="only CD features (train)")
        scatter_hard_sample(bestEst,pd.concat([X_test,y_test],axis=1),pipeIDM,prob=True,add_txt="only CD features (test)")
        fi=pd.DataFrame(bestEst.steps[-1][1].feature_importances_,index=["b_dist","m_dist"],columns=["Importance"])
        print(f"Feature importance:\n {fi.head(11)}")


        print("\n******Without extracted features********\n")
        _,_,_,est=train_test_model(pipeIRF,parameters,df_train,X_test,y_test,add_title="Random Forest without extracted features")#Without new features
        fi=pd.DataFrame(est.steps[-1][1].feature_importances_,index=X_train.columns,columns=["Importance"])
        print(f"Feature importance:\n {fi.head(9)}")

    if cmd=="svc" or cmd=="all":
        #SVC
        print("**************Support Vector Classifier*************************")
        svc=SVC(random_state=42,probability=True)
        tmpIDMS=list(pipeIDMS.steps) #Copy the full ipeline
        tmpIDMS.append(("svc",svc)) #Append the model

        tmpIDM=list(pipeIDM.steps) #Copy the full ipeline
        tmpIDM.append(("scaler2",scaler))
        tmpIDM.append(("svc",svc)) #Append the model

        tmpIM=list(pipeIM.steps)
        tmpIM.append(("svc",svc)) #Append the model




        pipeIDRFS=Pipeline(tmpIDMS) #Instntiate pipeline object
        pipeIDRF=Pipeline(tmpIDM)
        pipeIRF=Pipeline(tmpIM)
        
        parameters = {'svc__kernel':('linear', 'poly', 'rbf', 'sigmoid'),
                    'svc__C':(0.1,0.5,1.0,10,50),
                    'svc__gamma':('scale','auto') 
                    }
        #Train and test in cross validation grid search and test with test data
        print("\n******With extracted features and original********\n")
        _,_,_,bestEst=train_test_model(pipeIDRFS,parameters,df_train,X_test,y_test,add_title="SVC with extracted and original features")#With new features
        scatter_hard_sample(bestEst,df_train,pipeIDM,prob=True,add_txt="CD and original features (train)")
        scatter_hard_sample(bestEst,pd.concat([X_test,y_test],axis=1),pipeIDM,prob=True,add_txt="CD and original features (test)")
     

        print("\n******With only extracted features********\n")
        _,_,_,bestEst=train_test_model(pipeIDRF,parameters,df_train,X_test,y_test,add_title="SVC with extracted features")#With new features
        scatter_hard_sample(bestEst,df_train,pipeIDM,prob=True,add_txt="only CD features (train)")
        scatter_hard_sample(bestEst,pd.concat([X_test,y_test],axis=1),pipeIDM,prob=True,add_txt="only CD features (test)")
  

        print("\n******Without extracted features********\n")
        _,_,_,est=train_test_model(pipeIRF,parameters,df_train,X_test,y_test,add_title="SVC without extracted features")#Without new features
      
        
        

        
    
    if cmd=="lr" or cmd=="all":
        #Logistic Regression TO DO: il solver rbfg supporta solo l2 o None sistemare i grid parameter
        print("**************Logistic Regression*************************")
        lr=LogisticRegression(random_state=42)
        tmpIDMS=list(pipeIDMS.steps) #Copy the full ipeline
        tmpIDMS.append(("lr",lr)) #Append the model

        tmpIDM=list(pipeIDM.steps) #Copy the full ipeline
        tmpIDM.append(("lr",lr)) #Append the model

        tmpIM=list(pipeIM.steps)
        tmpIM.append(("lr",lr)) #Append the model




        pipeIDRFS=Pipeline(tmpIDMS) #Instntiate pipeline object
        pipeIDRF=Pipeline(tmpIDM)
        pipeIRF=Pipeline(tmpIM)
        
        parameters = {'lr__penalty':('none', 'l2'),
                    'lr__dual':([False]),
                    "lr__C": [1, 10, 100],
                    'lr__max_iter':[500,1000]

                    }
        #Train and test in cross validation grid search and test with test data
        print("\n******With extracted features and original********\n")
        _,_,_,bestEst=train_test_model(pipeIDRFS,parameters,df_train,X_test,y_test,add_title="Log Regression with extracted and original features")#With new features
        scatter_hard_sample(bestEst,df_train,pipeIDMS,prob=True,add_txt="CD and original features (train)")
        scatter_hard_sample(bestEst,pd.concat([X_test,y_test],axis=1),pipeIDMS,prob=True,add_txt="CD and original features (test)")
     

        print("\n******With only extracted features********\n")
        _,_,_,bestEst=train_test_model(pipeIDRF,parameters,df_train,X_test,y_test,add_title="Log Regression with extracted features")#With new features
        scatter_hard_sample(bestEst,df_train,pipeIDM,prob=True,add_txt="only CD features (train)")
        scatter_hard_sample(bestEst,pd.concat([X_test,y_test],axis=1),pipeIDM,prob=True,add_txt="only CD features (test)")
  

        print("\n******Without extracted features********\n")
        _,_,_,est=train_test_model(pipeIRF,parameters,df_train,X_test,y_test,add_title="Log Regression without extracted features")#Without new features
      
        
        

    if cmd=="knn" or cmd=="all":
        #K nearest neighboor
        print("**************KNN*************************")
        knn=KNN()
        tmpIDMS=list(pipeIDMS.steps) #Copy the full ipeline
        tmpIDMS.append(("knn",knn)) #Append the model

        tmpIDM=list(pipeIDM.steps) #Copy the full ipeline
        tmpIDM.append(("scaler 2",scaler)) #Append the model
        tmpIDM.append(("knn",knn)) #Append the model

        tmpIM=list(pipeIM.steps)
        tmpIM.append(("knn",knn)) #Append the model




        pipeIDRFS=Pipeline(tmpIDMS) #Instntiate pipeline object
        pipeIDRF=Pipeline(tmpIDM)
        pipeIRF=Pipeline(tmpIM)
        
        parameters = {'knn__n_neighbors':(1,2, 3,5,10),
                    'knn__weights':("uniform","distance"),
                    }
        
        #Train and test in cross validation grid search and test with test data
        print("\n******With extracted features and original********\n")
        _,_,_,bestEst=train_test_model(pipeIDRFS,parameters,df_train,X_test,y_test,add_title="KNN with extracted and original features")#With new features
        scatter_hard_sample(bestEst,df_train,pipeIDMS,prob=True,add_txt="CD and original features (train)")
        scatter_hard_sample(bestEst,pd.concat([X_test,y_test],axis=1),pipeIDMS,prob=True,add_txt="CD and original features (test)")
     

        print("\n******With only extracted features********\n")
        _,_,_,bestEst=train_test_model(pipeIDRF,parameters,df_train,X_test,y_test,add_title="KNN with extracted features")#With new features
        scatter_hard_sample(bestEst,df_train,pipeIDM,prob=True,add_txt="only CD features (train)")
        scatter_hard_sample(bestEst,pd.concat([X_test,y_test],axis=1),pipeIDM,prob=True,add_txt="only CD features (test)")
  

        print("\n******Without extracted features********\n")
        _,_,_,est=train_test_model(pipeIRF,parameters,df_train,X_test,y_test,add_title="KNN without extracted features")#Without new features
      
        
       

    if cmd=="mlp" or cmd=="all":
        #MLP
        print("**************Neural Net MLP************************")
        mlp=MLPClassifier(random_state=42)
        tmpIDM=list(pipeIDM.steps) #Copy the full ipeline
        #tmpIDM.append(("scaler",StandardScaler()))
        tmpIDM.append(("mlp",mlp)) #Append the model
        tmpIM=list(pipeIM.steps) #Same for the pipeline without feature extraction
        tmpIM.append(("mlp",mlp))
        pipeIDSVC=Pipeline(tmpIDM) #Instntiate pipeline object
        pipeISVC=Pipeline(tmpIM)
        parameters = {"mlp__max_iter":[200],
                      "mlp__hidden_layer_sizes":[(100,)]}
        #                "mlp__activation":['identity', 'logistic', 'tanh', 'relu']
        #            }
        #train_test_model(pipeIDSVC,parameters,df_train,X_test,y_test,add_title="MLP with extracted features")
        
        print("\n******Without extracted features********\n")
        _,_,_,bestEst=train_test_model(pipeISVC,parameters,df_train,X_test,y_test,add_title="MLP without extracted features")
        scatter_hard_sample(bestEst,pd.concat([X_test,y_test],axis=1),pipeIDM,prob=True)

   

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description='Analysis/Exploration of the dataset and training of the main ML model '
    )
    parser.add_argument('--data', required=True, help='The path of the file containing the dataset')
    parser.add_argument('--run_analysis',default='all',help="The analysis to run: [no]: skip the analysis [ext]: scatter of 2 extracted features [pca]: run only pca analysis [correlation]: run only correlation analysis [all]: run all analysis")
    parser.add_argument('--run_training',default='all',help="The model to train: [no]: skip the training [rf]: random forest [svc]: support vector machine [lr]: logistic regression [knn]: k nearest neighboors [all]: all models")
    parser.add_argument('--test_size',default=0.33,help="The percentage of the test set")
    args = parser.parse_args()
    print("*********LOADING, BASIC CLEANING & SPLITTING***************")
    #LOADING & BASIC CLEANING
    df=load_dataset(args.data)#Load dataset
    print(df)
    df=basic_clean(df)#Drop duplicates check for null and change null ('?') to np.NaN
    
    #DATASET SPLITTING (TRAIN-TEST)
    X_train,X_test,y_train,y_test=prepare_training(df,test_size=args.test_size)#Split into train and test
    df_train=pd.concat([X_train,y_train],axis=1)#Instantiate dataframe of training data
    
   

    #Prepro pipeline
    imp=SimpleImputer(missing_values=np.NaN,strategy="median")#Replace NaN with median
    ext=DistExtractor(onlyCD=True)#Extract centroids and retrieve only 2 feature distance for beningn centroids and malign centroids
    extAll=DistExtractor(onlyCD=False)
    scaler=StandardScaler()
    dfbuilder=DfBuilder()
    
    pipeIDM=Pipeline(steps=[("Imputer",imp),("Standard scaler",scaler),("Distance extractor",ext)])#with new feature extracted
    pipeIM=Pipeline(steps=[("Imputer",imp),("Standard scaler",scaler)])#Without feature extracted
    pipeIDMS=Pipeline(steps=[("Imputer",imp),("Standard scaler",scaler),("Distance extractor",extAll),("Standard scaler2",scaler),("DF builder",dfbuilder)])
    #pipeIDM=Pipeline(steps=[("Imputer",imp),("Distance extractor",ext)])#with new feature extracted
    #pipeIM=Pipeline(steps=[("Imputer",imp)])#Without feature extracted

    cols=pipeIDMS.fit_transform(X_train,y_train).columns#get the columns after preprocessing (to printo below)

    #Data analysis and exploration
    if args.run_analysis!='no':
        data_exploration(X_train,y_train,imp,extAll,scaler,args.run_analysis)
        

    if args.run_training!='no':
        training(pipeIDM,pipeIM,pipeIDMS,df_train,X_test,y_test,args.run_training)
       
    plt.show()
    exit()

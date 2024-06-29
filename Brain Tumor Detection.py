import warnings
warnings.filterwarnings(action="ignore")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.model_selection import KFold
from sklearn.naive_bayes import GaussianNB
import time
pd.set_option('display.width',1000)
pd.set_option('display.max_columns',None)
df=pd.read_csv("D:\BrainTumorData.csv")
del df['Unnamed: 32']
df['diagnosis']=df['diagnosis'].apply(lambda x:1 if x=='M' else 0)
df=df.set_index('id')
#print(df.head(15))
x=df.drop('diagnosis',axis=1).values
y=df['diagnosis'].values
df.plot(kind='density',layout=(5,7),subplots=True)
xtrain,xtest,ytrain,ytest=train_test_split(x,y,shuffle=True,random_state=123,test_size=0.33)
kfold=KFold(n_splits=10,shuffle=True,random_state=21)
pipeline=[]
pipeline.append(('Scaled LR',Pipeline([('Scaler',StandardScaler()),('LR',LogisticRegression())])))
pipeline.append(('Scaled KNN',Pipeline([('Scaler',StandardScaler()),('KNN',KNeighborsClassifier())])))
pipeline.append(('Scaled Decision Trees',Pipeline([('Scaler',StandardScaler()),('DT',DecisionTreeClassifier())])))
pipeline.append(('Scaled RFA',Pipeline([('Scaler',StandardScaler()),('RFA',RandomForestClassifier())])))
pipeline.append(('Scaled LDA',Pipeline([('Scaler',StandardScaler()),('LDA',LinearDiscriminantAnalysis())])))
pipeline.append(('Scaled SVM',Pipeline([('Scaler',StandardScaler()),('SVM',SVC())])))
pipeline.append(('Scaled NB',Pipeline([('Scaler',StandardScaler()),('NB',GaussianNB())])))
results=[]
for name,model in pipeline:
    start=time.time()
    result=cross_val_score(model,xtrain,ytrain,cv=kfold,scoring='accuracy')
    end=time.time()
    print(name,result.mean()*100,end-start)









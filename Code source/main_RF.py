



from sklearn.model_selection import KFold, cross_val_score as CVS, train_test_split as TTS
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
from sklearn.utils import shuffle
from xgboost import XGBClassifier as XGBC
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import NearMiss
from sklearn.metrics import silhouette_score
import time
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
import seaborn as sns 
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn import svm
from matplotlib.ticker import MultipleLocator
from sklearn.linear_model import LogisticRegression

n=100



setting_= pd.read_excel(r"C:/Users/86362/Desktop/granite training set.xlsx")
setting_=setting_.dropna(inplace=False)
setting_= shuffle(setting_,random_state=n)
#setting_=setting_.drop(labels=['U/Yb','Eu/Gd'],axis=1)
X=setting_.iloc[:,1:]
y=setting_['label']
    

# =============================================================================
# X_resampled, y_resampled = SMOTE(random_state=n,sampling_strategy={0:1000,1:1000,2:1000,3:1000,4:1000,5:1000,6:1000}).fit_resample(X, y)
# X_resampled, y_resampled = NearMiss(version=2,sampling_strategy={0:600,1:600,2:600,3:600,4:600,5:600,6:600}).fit_resample(X_resampled, y_resampled)
# cc=pd.concat([y_resampled,X_resampled],axis=1)
# setting_= shuffle(cc,random_state=n)
# X=setting_.iloc[:,1:]
# y=setting_['label']
# =============================================================================

# =============================================================================
# uu=pd.DataFrame(columns=['A'])
# for i in range(18):
#     for j in range(i,18):
#         uu[X.columns.values[i]+'/'+X.columns.values[j+1]]=X.iloc[:,i]/X.iloc[:,j+1]
# 
# cc=pd.concat([uu,X],axis=1)
# X=cc.drop(['A'],axis=1)
# X=np.log10(X)
# 
# X=pd.concat([y,X],axis=1)
# X.to_excel(r"C:/Users/86362/Desktop/predicting set.xlsx")
# =============================================================================

# =============================================================================
# cc=pd.concat([X['P*'],X['REE+Y']],axis=1)
# cc=pd.concat([y,cc],axis=1)
# KF = KFold(n_splits=5,shuffle=True,random_state=100)
# 
# XG=XGBC(n_estimators=100,learning_rate=0.1,max_depth=6,random_state=100)
# RFclassifier  = RandomForestClassifier(n_estimators = 100,random_state = 100)
# svmc  = SVC(probability=True)
# svmc=svm.LinearSVC(random_state=100,max_iter=1000)
# knn=KNeighborsClassifier()
# ann=MLPClassifier((100,100),max_iter=1000)
# print((cross_val_score(svmc,cc.iloc[:,1:],cc.iloc[:,0],scoring='accuracy',cv=5)).mean())
# =============================================================================




Xpridict=pd.read_excel(r"C:/Users/86362/Desktop/jack hill(P).xlsx")
Xpridict=Xpridict.iloc[:,2:23]
#Xpridict=Xpridict.drop(labels=['Ti'],axis=1)
uu=pd.DataFrame(columns=['A'])
# =============================================================================
# for i in range(18):
#     for j in range(i,18):
#         uu[Xpridict.columns.values[i]+'/'+Xpridict.columns.values[j+1]]=Xpridict.iloc[:,i]/Xpridict.iloc[:,j+1]
# =============================================================================

cc=pd.concat([uu,Xpridict],axis=1)
Xpridict=cc.drop(['A'],axis=1)
Xpridict=np.log10(Xpridict)

Xpridict=pd.concat([Xpridict['Ce'],Xpridict['Eu']],axis=1)
X = pd.concat([X['Ce'],X['Eu']],axis=1) 
reg = RandomForestClassifier(n_estimators = 100,random_state = 100).fit(X,y)
ypridict=reg.predict(Xpridict)
ypridict=pd.DataFrame(ypridict)
ypridict.to_excel(r"C:/Users/86362/Desktop/predicting set.xlsx")


# =============================================================================
# model =RandomForestClassifier(n_estimators = 100,random_state =100).fit(X,y)
# explainer = shap.explainers.Tree(model)
# shap_values = explainer.shap_values(X)
# 
# shap.summary_plot(shap_values, X, plot_type="bar")
# plt.show()
# =============================================================================

# =============================================================================
# setting_= pd.read_excel(r"C:/Users/86362/Desktop/granite.xlsx")
# X=setting_.iloc[:,1:]
# y=setting_['label']
# JH= pd.read_excel(r"C:/Users/86362/Desktop/jack hill.xlsx")
# TTG= pd.read_excel(r"C:/Users/86362/Desktop/SPGs.xlsx")
# 
# y_major_Locator=MultipleLocator(1)
# 
# 
# X=pd.concat([y,X],axis=1)
# A=X.loc[X['label'] ==0]
# B=X.loc[X['label'] ==1]
# 
# 
# 
# g = sns.JointGrid()
# g.figure.set_size_inches(12,12)
# sns.kdeplot(x=A['Eu/Eu*'], linewidth=1.5, ax=g.ax_marg_x, fill=True,color='red')
# sns.kdeplot(x=B['Eu/Eu*'], linewidth=1.5, ax=g.ax_marg_x, fill=True,color='blue')
# 
# 
# 
# sns.kdeplot(y=A['P/Tm'], linewidth=1.5, ax=g.ax_marg_y, fill=True,color='red')
# sns.kdeplot(y=B['P/Tm'], linewidth=1.5, ax=g.ax_marg_y, fill=True,color='blue')
# 
# 
# 
# 
# 
# g.ax_joint.scatter(A['Eu/Eu*'],A['P/Tm'],label="I type",s=100,marker='o',c='red')
# g.ax_joint.scatter(B['Eu/Eu*'],B['P/Tm'],label="S type",s=100,marker='o',c='blue')
# g.ax_joint.scatter(JH['Eu/Eu*'],JH['P/Tm'],label="Jack Hill",s=100,marker='s',c='deepskyblue')
# g.ax_joint.scatter(TTG['Eu/Eu*'],TTG['P/Tm'],label="SPGs",s=100,marker='s',c='tomato')
# 
# sns.kdeplot(x=A['Eu/Eu*'],y=A['P/Tm'],
#   levels=1, thresh=.15,ax=g.ax_joint,color='red',linewidths=5)
# sns.kdeplot(x=B['Eu/Eu*'],y=B['P/Tm'],
#   levels=1, thresh=.15,ax=g.ax_joint,color='blue',linewidths=5)
# 
# 
# g.ax_joint.legend(loc='upper left',ncol=2,fontsize=20)
# g.ax_joint.set_xlabel("log$_{10}$ Eu/Eu*",fontsize=30)
# g.ax_joint.set_ylabel("log$_{10}$ P/Tm",fontsize=30)
# g.ax_joint.tick_params(labelsize=30)
# =============================================================================

















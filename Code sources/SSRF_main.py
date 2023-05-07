import numpy as np
import matplotlib.pyplot as plt
from sklearn.semi_supervised import SelfTrainingClassifier
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import NearMiss
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
import pandas as pd
import shap
from sklearn.utils import shuffle

def getdata_2settings():
    unlabel = pd.read_csv("unlabel 2 settings.csv")
    unlabel_y = unlabel.iloc[:,0]
    unlabel_x = unlabel.iloc[:,2:]
    unlabel_x.loc[unlabel_x['Eu'] >=0,'Eu'] = unlabel_x['Eu']/0.0563/np.sqrt(unlabel_x['Sm']/0.148*unlabel_x['Gd']/0.199)
    unlabel_x=np.log10(unlabel_x)
    unlabel_x.rename(columns={'Eu':'Eu/Eu*'},inplace=True) 
    label = pd.read_csv("label 2 settings.csv")
    label=shuffle(label,random_state=45)
    label_y = label.iloc[:,0]
    label_x = label.iloc[:,2:]
    label_x.loc[label_x['Eu'] >=0,'Eu'] = label_x['Eu']/0.0563/np.sqrt(label_x['Sm']/0.148*label_x['Gd']/0.199)
    label_x=np.log10(label_x)
    label_x.rename(columns={'Eu':'Eu/Eu*'},inplace=True)
    label=pd.concat([label_y,label_x],axis=1)
    return unlabel_x, unlabel_y, label_x, label_y

def getdata_7settings():
    unlabel = pd.read_csv("unlabel 7 settings.csv")
    unlabel_y = unlabel.iloc[:,0]
    unlabel_x = unlabel.iloc[:,2:]
    unlabel_x.loc[unlabel_x['Eu'] >=0,'Eu'] = unlabel_x['Eu']/0.0563/np.sqrt(unlabel_x['Sm']/0.148*unlabel_x['Gd']/0.199)
    unlabel_x=np.log10(unlabel_x)
    unlabel_x.rename(columns={'Eu':'Eu/Eu*'},inplace=True) 
    label = pd.read_csv("label 7 settings.csv")
    label=shuffle(label,random_state=45)
    label_y = label.iloc[:,0]
    label_x = label.iloc[:,2:]
    label_x.loc[label_x['Eu'] >=0,'Eu'] = label_x['Eu']/0.0563/np.sqrt(label_x['Sm']/0.148*label_x['Gd']/0.199)
    label_x=np.log10(label_x)
    label_x.rename(columns={'Eu':'Eu/Eu*'},inplace=True)
    label=pd.concat([label_y,label_x],axis=1)
    smote_x, smote_y = SMOTE(random_state=45
                             ,sampling_strategy={0:1000,1:1000,2:1000,3:1000,4:1000,5:1000,6:1000}).fit_resample(label_x, label_y)
    resample_x, resample_y = NearMiss(version=2
                                      ,sampling_strategy={0:600,1:600,2:600,3:600,4:600,5:600,6:600}).fit_resample(smote_x, smote_y)
    label_x= shuffle(resample_x,random_state=45)
    label_y= shuffle(resample_y,random_state=45)
    return unlabel_x, unlabel_y, label_x, label_y

def getdata_gra():
    unlabel = pd.read_csv("unlabel granite.csv")
    unlabel_y = unlabel.iloc[:,0]
    unlabel_x = unlabel.iloc[:,2:]
    unlabel_x.loc[unlabel_x['Eu'] >=0,'Eu'] = unlabel_x['Eu']/0.0563/np.sqrt(unlabel_x['Sm']/0.148*unlabel_x['Gd']/0.199)
    unlabel_x=np.log10(unlabel_x)
    unlabel_x.rename(columns={'Eu':'Eu/Eu*'},inplace=True) 
    label = pd.read_csv("label granite.csv")
    label= shuffle(label,random_state=45)
    label_y = label.iloc[:,0]
    label_x = label.iloc[:,2:]
    label_x.loc[label_x['Eu'] >=0,'Eu'] = label_x['Eu']/0.0563/np.sqrt(label_x['Sm']/0.148*label_x['Gd']/0.199)
    label_x=np.log10(label_x)
    label_x.rename(columns={'Eu':'Eu/Eu*'},inplace=True)
    label=pd.concat([label_y,label_x],axis=1)
    label_x, label_y = SMOTE(random_state=45
                             ,sampling_strategy={0:300,1:300}
                            ).fit_resample(label_x, label_y)
    label_x= shuffle(label_x,random_state=45)
    label_y= shuffle(label_y,random_state=45)    
    return unlabel_x, unlabel_y, label_x, label_y

def savedata(predict_x, Y_hat):
    predict = pd.read_csv("δ18O.csv")
    predict.insert(predict.shape[1],'label',Y_hat)
    predict.to_excel("result_δ18O.xlsx")

def plot_featureimportance(model,X):
    c = X.columns.values
    explainer = shap.explainers.Tree(model)
    shap_values = explainer.shap_values(X)
    shap.summary_plot(shap_values, c, plot_type="bar")
    plt.show()

#data
unlabel_x, unlabel_y,label_x, label_y = getdata_gra()
label_xx = np.concatenate((label_x,unlabel_x),axis=0)
label_yy = np.concatenate((label_y,unlabel_y),axis=0)

# feature importance
base_classifier = RandomForestClassifier(n_estimators=100, random_state=100)
self_training_model = SelfTrainingClassifier(base_estimator=base_classifier, threshold=0.8,criterion='threshold',max_iter=20,verbose=True)
self_training = self_training_model.fit(label_xx, label_yy)
plot_featureimportance(self_training.base_estimator_,label_x)

# predict
predict = pd.read_csv("δ18O.csv")
predict = predict.iloc[:,2:]
predict.loc[predict['Eu'] >=0,'Eu'] = predict['Eu']/0.0563/np.sqrt(predict['Sm']/0.148*predict['Gd']/0.199)
predict=np.log10(predict)
Y_hat = self_training.predict(predict)
savedata(predict,Y_hat)

# evaluation
# =============================================================================
# x_train, x_test, y_train, y_test = train_test_split(label_x, label_y, test_size = 0.3, random_state = 100)
# label_xx = np.concatenate((x_train,unlabel_x),axis=0)
# label_yy = np.concatenate((y_train,unlabel_y),axis=0)
# base_classifier = RandomForestClassifier(n_estimators=100, random_state=100)
# self_training_model = SelfTrainingClassifier(base_estimator=base_classifier, threshold=0.8,criterion='threshold',max_iter=20,verbose=True)
# self_training = self_training_model.fit(label_xx, label_yy)
# Acu = self_training.score(x_test, y_test)
# print("accuracy : ", Acu)
# Pre = precision_score(y_test, self_training.predict(x_test)
#                       , average='micro'
#                       )
# print("precision : ", Pre)
# reca = recall_score(y_test, self_training.predict(x_test)
#                     , average='micro'
#                     )
# print("recall : ", reca)
# f1 = f1_score(y_test, self_training.predict(x_test)
#               , average='micro'
#               )
# print("f1 : ", f1)
# AUC = roc_auc_score(y_test, self_training.predict(x_test)
#                     ,multi_class='ovo'
#                     )
# print("AUC : ", AUC)
# =============================================================================


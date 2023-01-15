import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.semi_supervised import SelfTrainingClassifier
from sklearn.metrics import classification_report
from imblearn.over_sampling import RandomOverSampler,SMOTE,ADASYN
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
import pandas as pd
import shap

def getdata():
    unlabel = pd.read_csv("unlabel777.csv")
    unlabel_y = unlabel.iloc[:,0].values
    unlabel_x = unlabel.iloc[:,1:].values
    label = pd.read_csv("label777.csv")
    label_y = label.iloc[:,0].values
    label_x = label.iloc[:,1:].values
    predictttg = pd.read_csv("predictttg777.csv")
    predict_ttg = predictttg.values
    predictjh = pd.read_csv("predictjh777.csv")
    predict_jh = predictjh.values
    predictluzhi = pd.read_csv("predictluzhi_hua.csv")
    predict_luzhi = predictluzhi.values
    return unlabel_x, unlabel_y, label_x, label_y, predict_ttg, predict_jh,predict_luzhi

def savedata_ttg(predict_x, Y_hat, Y_proba):
    proba0 = Y_proba[:,0]
    proba1 = Y_proba[:,1]
    proba2 = Y_proba[:,2]
    proba3 = Y_proba[:,3]
    proba4 = Y_proba[:,4]
    proba5 = Y_proba[:,5]
    proba6 = Y_proba[:,6]
    # print(proba0)
    print(proba0.shape)
    predict = pd.read_csv("predictttg777.csv")
    predict.insert(predict.shape[1],'proba0',proba0)
    predict.insert(predict.shape[1],'proba1',proba1)
    predict.insert(predict.shape[1],'proba2',proba2)
    predict.insert(predict.shape[1],'proba3',proba3)
    predict.insert(predict.shape[1],'proba4',proba4)
    predict.insert(predict.shape[1],'proba5',proba5)
    predict.insert(predict.shape[1],'proba6',proba6)
    predict.insert(predict.shape[1],'label',Y_hat)
    # print(predict['proba0'])
    predict.to_excel("result_ttg777.xlsx")
    
def savedata_jh(predict_x, Y_hat, Y_proba):
    proba0 = Y_proba[:,0]
    proba1 = Y_proba[:,1]
    proba2 = Y_proba[:,2]
    proba3 = Y_proba[:,3]
    proba4 = Y_proba[:,4]
    proba5 = Y_proba[:,5]
    proba6 = Y_proba[:,6]
    # print(proba0)
    print(proba0.shape)
    predict = pd.read_csv("predictjh777.csv")
    predict.insert(predict.shape[1],'proba0',proba0)
    predict.insert(predict.shape[1],'proba1',proba1)
    predict.insert(predict.shape[1],'proba2',proba2)
    predict.insert(predict.shape[1],'proba3',proba3)
    predict.insert(predict.shape[1],'proba4',proba4)
    predict.insert(predict.shape[1],'proba5',proba5)
    predict.insert(predict.shape[1],'proba6',proba6)
    predict.insert(predict.shape[1],'label',Y_hat)
    # print(predict['proba0'])
    predict.to_excel("result_jh777.xlsx")

def savedata_luzhi(predict_x, Y_hat, Y_proba):
    proba0 = Y_proba[:,0]
    proba1 = Y_proba[:,1]
    # print(proba0)
    print(proba0.shape)
    predict = pd.read_csv("predictluzhi_hua.csv")
    predict.insert(predict.shape[1],'proba0',proba0)
    predict.insert(predict.shape[1],'proba1',proba1)
    predict.insert(predict.shape[1],'label',Y_hat)
    # print(predict['proba0'])
    predict.to_excel("result_luzhi_hua.xlsx")

def plot_featureimportance(model,X):
    predict = pd.read_csv("predictjh777.csv")
    c = predict.columns.values
    explainer = shap.explainers.Tree(model)
    shap_values = explainer.shap_values(X)
    shap.summary_plot(shap_values, c, plot_type="bar")
    plt.show()


# data
unlabel_x, unlabel_y,label_x, label_y, predict_ttg, predict_jh, predict_luzhi = getdata()
# smote_x, smote_y = SMOTE().fit_resample(label_x, label_y)
train_x = np.concatenate((label_x,unlabel_x),axis=0)
train_y = np.concatenate((label_y,unlabel_y),axis=0)
# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 0)

# train
base_classifier = RandomForestClassifier(n_estimators=100, random_state=100)
self_training_model = SelfTrainingClassifier(base_estimator=base_classifier, threshold=0.8,criterion='threshold',max_iter=20,verbose=True)
self_training = self_training_model.fit(train_x, train_y)
# evaluation
accuracy_score_ST = self_training.score(label_x, label_y)
print('Accuracy Score: ', accuracy_score_ST)
# print(classification_report(label_y, self_training.predict(label_x)))
# Pre = precision_score(y_test, self_training.predict(x_test), average='micro')
# print("precision : ", Pre)
# reca = recall_score(y_test, self_training.predict(x_test), average='micro')
# print("recall : ", reca)
# f1 = f1_score(y_test, self_training.predict(x_test), average='micro')
# print("f1 : ", f1)
# AUC = roc_auc_score(y_test, self_training.predict_proba(x_test),multi_class='ovo')
# # AUC = roc_auc_score(y_test, self_training.predict_proba(x_test)[:,1])
# print("AUC : ", AUC)
# predict
Y_hat = self_training.predict(predict_ttg)
Y_proba = self_training.predict_proba(predict_ttg)
savedata_ttg(predict_ttg,Y_hat,Y_proba)
Y_hat = self_training.predict(predict_jh)
Y_proba = self_training.predict_proba(predict_jh)
savedata_jh(predict_jh,Y_hat,Y_proba)
# Y_hat = self_training.predict(predict_luzhi)
# Y_proba = self_training.predict_proba(predict_luzhi)
# savedata_luzhi(predict_jh,Y_hat,Y_proba)

plot_featureimportance(self_training.base_estimator_,label_x)
print(self_training.base_estimator_.feature_importances_)
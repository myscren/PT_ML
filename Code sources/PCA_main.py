import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from matplotlib.ticker import MultipleLocator
from sklearn.svm import SVC

plt.rcParams['font.family'] = ['serif']
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']

# data
data1=pd.read_csv("label 2 settings.csv")
data1.loc[data1['Eu'] >=0,'Eu'] = data1['Eu']/0.0563/np.sqrt(data1['Sm']/0.148*data1['Gd']/0.199)
data1.rename(columns={'Eu':'Eu/Eu*'},inplace=True) 
data2=pd.read_csv("unlabel 2 settings.csv")
data2.loc[data2['Eu'] >=0,'Eu'] = data2['Eu']/0.0563/np.sqrt(data2['Sm']/0.148*data2['Gd']/0.199)
data2.rename(columns={'Eu':'Eu/Eu*'},inplace=True) 
datax=pd.concat([data1,data2],axis=0)
datax=datax.iloc[:,2:]
datax=np.log10(datax)

# Standardization
datax=(datax-np.mean(datax,axis=0))/np.std(datax,axis=0)
pca=PCA(n_components=2,whiten=False,svd_solver='auto',tol=0.1)
pca1=pca.fit(datax)
pca_var1=pca1.explained_variance_ratio_
pca_compo=pd.DataFrame(np.around(pca1.components_.T,3))


# PCA plot
# =============================================================================
# plt.figure(figsize=(10,10))
# 
# for i in range (17):
#     plt.text(pca_compo.iloc[i,0]*1.05,pca_compo.iloc[i,1]*1.05, data1.iloc[:,2:].columns[i]
#              ,fontsize=20)
#     plt.arrow(0,0,pca_compo.iloc[i,0]*0.95,pca_compo.iloc[i,1]*0.95,width=0.001,length_includes_head=True
#               ,head_width=0.02)
# 
# plt.xlabel("PC 1 (59.31%)",fontsize=25)
# plt.ylabel("PC 2 (10.78%)",fontsize=25)
# plt.tick_params(labelsize=25)
# plt.xlim(-0.4,0.3)
# plt.ylim(-0.6,0.5)
# plt.show()
# =============================================================================

# kernel plot

# =============================================================================
# jack=pd.read_csv("Jack Hill settings.csv")
# ttg=pd.read_csv("TTG settings.csv")
# 
# dataxj=jack.iloc[:,1:]
# dataxj.loc[dataxj['Eu'] >=0,'Eu'] = dataxj['Eu']/0.0563/np.sqrt(dataxj['Sm']/0.148*dataxj['Gd']/0.199)
# dataxj.rename(columns={'Eu':'Eu/Eu*'},inplace=True) 
# dataxj=np.log10(dataxj)
# 
# 
# dataxt=ttg.iloc[:,1:]
# dataxt.loc[dataxt['Eu'] >=0,'Eu'] = dataxt['Eu']/0.0563/np.sqrt(dataxt['Sm']/0.148*dataxt['Gd']/0.199)
# dataxt.rename(columns={'Eu':'Eu/Eu*'},inplace=True) 
# dataxt=np.log10(dataxt)
# 
# 
# 
# con=data1.loc[data1['label'] ==0]
# oce=data1.loc[data1['label'] ==1]
# con=con.iloc[:,2:]
# oce=oce.iloc[:,2:]
# con=np.log10(con)
# oce=np.log10(oce)
# 
# g = sns.JointGrid()
# g.figure.set_size_inches(10,10)
# y_major_Locator=MultipleLocator(1)
# sns.kdeplot(x=np.dot(oce,pca_compo[0]), linewidth=1.5, ax=g.ax_marg_x, fill=True,color='red')
# sns.kdeplot(x=np.dot(con,pca_compo[0]), linewidth=1.5, ax=g.ax_marg_x, fill=True,color='blue')
# 
# sns.kdeplot(y=np.dot(oce,pca_compo[1]), linewidth=1.5, ax=g.ax_marg_y, fill=True,color='red')
# sns.kdeplot(y=np.dot(con,pca_compo[1]), linewidth=1.5, ax=g.ax_marg_y, fill=True,color='blue')
# 
# g.ax_joint.scatter(np.dot(con,pca_compo[0]),np.dot(con,pca_compo[1]),label="Continetal (Phanerozoic)",s=100,marker='o',c='red') 
# g.ax_joint.scatter(np.dot(oce,pca_compo[0]),np.dot(oce,pca_compo[1]),label="Oceanic (Phanerozoic)",s=100,marker='o',c='blue')
# g.ax_joint.scatter(np.dot(dataxj,pca_compo[0]),np.dot(dataxj,pca_compo[1]),label="Jack Hill",s=100,marker='^',c='yellow', edgecolor='black') 
# g.ax_joint.scatter(np.dot(dataxt,pca_compo[0]),np.dot(dataxt,pca_compo[1]),label="TTG",s=100,marker='^',c='green', edgecolor='black') 
# 
# sns.kdeplot(x=np.dot(oce,pca_compo[0]),y=np.dot(oce,pca_compo[1]),
#    levels=1, thresh=.15,ax=g.ax_joint,color='blue',linewidths=5)
# sns.kdeplot(x=np.dot(con,pca_compo[0]),y=np.dot(con,pca_compo[1]),
#    levels=1, thresh=.15,ax=g.ax_joint,color='red',linewidths=5)
# 
# g.ax_joint.legend(loc='lower right',ncol=1,fontsize=15)
# g.ax_joint.set_xlabel("PC 1 (59.31%)",fontsize=25)
# g.ax_joint.set_ylabel("PC 2 (10.78%)",fontsize=25)
# g.ax_joint.tick_params(labelsize=25)
# plt.show()
# =============================================================================


# decision boundary plot
# =============================================================================
# def plot_decision_boundary(clf , axes):
#     xp=np.linspace(axes[0], axes[1], 300)
#     yp=np.linspace(axes[2], axes[3], 300)
#     x1, y1=np.meshgrid(xp, yp) 
#     xy=np.c_[x1.ravel(), y1.ravel()] 
#     y_pred = clf.predict(xy).reshape(x1.shape) 
#     plt.contourf(x1, y1, y_pred, alpha=0.5, cmap=plt.cm.rainbow)
# 
# fig=plt.figure(figsize=[10,10])
# 
# 
# X=data1.iloc[:,2:]
# X=np.log10(X)
# y=data1.iloc[:,0]
# 
# 
# jack=pd.read_csv("Jack Hill settings.csv")
# 
# dataxj=jack.iloc[:,1:]
# dataxj.loc[dataxj['Eu'] >=0,'Eu'] = dataxj['Eu']/0.0563/np.sqrt(dataxj['Sm']/0.148*dataxj['Gd']/0.199)
# dataxj.rename(columns={'Eu':'Eu/Eu*'},inplace=True) 
# dataxj=np.log10(dataxj)
# 
# base_classifier = SVC(kernel="linear",random_state=100,probability=True)
# 
# clf=base_classifier.fit(pd.concat([X['U']-X['Tm'],X['Er']],axis=1), y)
# 
# plot_decision_boundary(clf, axes=[min(X['U']-X['Tm'])-0.4, max(X['U']-X['Tm'])+0.2, 
#                                   min(X['Er'])-0.5, max(X['Er'])+0.2])
# 
# continental=data1.loc[data1['label'] ==0]
# oceanic=data1.loc[data1['label'] ==1]
# continental=continental.iloc[:,2:]
# oceanic=oceanic.iloc[:,2:]
# continental=np.log10(continental)
# oceanic=np.log10(oceanic)
# 
# plt.scatter(continental['U']-continental['Tm'],continental['Er'],label="Continental",s=80,marker='o',c='black')
# plt.scatter(oceanic['U']-oceanic['Tm'],oceanic['Er'],label="Oceanic",s=80,marker='o',c='white', edgecolor='black')
# plt.scatter(dataxj['U']-dataxj['Tm'],dataxj['Er'],label="Jack Hill",s=80,marker='^',c='yellow', edgecolor='black')
# 
# plt.tick_params(labelsize=30)
# plt.legend(fontsize=20,ncol=1,loc='lower right')
# plt.xlabel("log$_{10}$ U/Tm",fontsize=30)
# plt.ylabel("log$_{10}$ Er",fontsize=30)
# =============================================================================


# PT_ML
## Overview
This repository contains data source and codes used in the paper entitled "Hadean tectonics: Insights from machine learning" by G. Chen et al. We proposed a novel independent chemical ‘fingerprinting’ approach based on semi-supervised machine learning (ML) with zircon trace element (TE) data (spanning 19 elements over the past 4.0 Gyrs) to accurately determine the tectono-magmatic provenances of zircons. The method was applied to Jack Hill zircons (JHZs) to improve insights into early crust construction. Our result provides clear evidence of sediment recycling associated with subduction activity in the Hadean. We compiled 3 data sources and 3 codes to generate the diagrams in Fig. 1-Fig. 4 and supplementary figures. Here is the list of data and code files.
## Data files
* [Data source](https://github.com/myscren/PT_ML/tree/main/Data%20source/) -- This file contains training datset and prediction dataset for ML. We combined recently compiled zircon TE chemistry data (P, Ti, Hf, Th, U, Y, and REEs [excluding Pr]) from previous publications (e.g., Borisova et al., 2022; Carley et al., 2022; Cavosie et al., 2006; Tang et al., 2021; references therein). Samples with La >1 ppm, (Dy/Nd) + (Dy/Sm) ≤ 30, and outliers with >mean +3SD were filtered in order to eliminate the altered zircons and statistical deviation of the compiled geochemical database, and ~6000 zircon analyses were used in this study. This database contians three training dataset used for ML or semi-supervised ML for distinguishing zircons formed in continental and oceanic crust, seven detailed tectonic environments, and S and I type granites. Also, this file contains the ML-based discriminant results of zircons provenances from TTG and Jack Hill samples.
## Code files
* [PCA_main.py](https://github.com/myscren/PT_ML/tree/main/Code%20source/PCA_main.py) -- Principle component analysis of zircon TE chemistry data. This program calcuates loading plot of elements and score plot of samples labeled by tectonic environments and granite types. 
* [RF_main.py](https://github.com/myscren/PT_ML/tree/main/Code%20source/RF.py) -- Tectono-magmatic provenances analysis of zircons using Random Forests (RF) based on zircon TE chemistry data. Please note that other ML algorithms, including Support Vector Machine (SVM), and Artificial Neural Network (ANN), XGBoost, K-NearestNeighbor (KNN), were implemented through this code file. One need to change the code block accordingly when runing these different ML algorithms.
* [SSRF_main.py](https://github.com/myscren/PT_ML/tree/main/Code%20source/SSRF.py) -- Tectono-magmatic provenances analysis of zircons using Semi-supervised Random Forests (SSRF) based on zircon TE chemistry data.
## User guide
To get a local copy up and running, please follow these simple steps.
### Prerequisites
The codes used in this paper were compiled on the Python 3.8.The versions of other packages are, specifically:
```
pandas == 1.4.2
numpy == 1.21.5
matplotlib == 3.5.1
seaborn == 0.11.2
sklearn == 1.1.3
shap == 0.41.0
...
```
## Installation Guide
The py codes require the Python complier installed on the PC or Laptop computer. To use the included PCA_main.py, RF_main.py and SSRF_main.py codes, one must install the sklearn, matplotlib, pandas and other basic function libraries on your Python complier. More details for installation and instruction of sklearn can be found at https://scikit-learn.org/stable/install.html.
### Download from Github
```
git clone https://github.com/myscren/PT_ML.git
cd PT_ML
```
### Running the codes

The python codes require the Python complier installed on the PC or Laptop computer. To use the [related codes](#code-files), one must install the related  [Prerequisites](#prerequisites) on your Python complier.

* Update the path of [data files](#data-files) in python codes
* Run the python codes in [code files](#code-files)

Example:
run PCA_main.py:
```
python PCA_main.py
```
run RF_main.py:
```
python RF_main.py
```
run SSRF_main.py:
```
python SSRF_main.py
```
## Authors

* **G. Chen** - *first author* - [GUOXIONG CHEN](https://grzy.cug.edu.cn/chenguoxiong)

To see more about the contributors to this project, please read the [paper](https://XXXXXX).

## License

PT_ML is licensed under the MIT License. See the LICENSE file for more details.

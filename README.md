# PT_ML
## Overview
This repository contains data source and codes used in the paper entitled "Hadean tectonics: Insights from machine learning" by G. Chen et al. We developed high-dimensional machine learning (ML) approaches using zircon chemistry data (spanning 19 elements over 4.0 b.y.) to characterize zircons that crystallized in some typical tectonic settings (e.g., arcs, plumerelated hotspots, and rifts) and from either igneous (I-type) or sedimentary (S-type) magmas.
## Data files
* [Data](/data/) -- We combined recently compiled zircon TE chemistry data (P, Ti, Hf, Th, U, Y, and REEs [excluding Pr]) from previous publications (e.g., Borisova et al., 2022; Carley et al., 2022; Cavosie et al., 2006; Tang et al., 2021; references therein). Samples with La >1 ppm, (Dy/Nd) + (Dy/Sm) ≤ 30, and outliers with >mean +3*δ were filtered in order to eliminate the altered zircons and statistical deviation of the compiled geochemical database, and ~6000 zircon analyses were used in this study. 
## Code files
* [rf.py](/codes/rf.py) -- Predicting tectonic environment and granite type using Random Forests with zircon TE chemistry data.
* [semi_rf.py](/codes/semi_rf) -- Predicting tectonic environment and granite type using semi-supervised Random Forests with zircon TE chemistry data.
* [PCa,kernel,decision_boundary.py](/codes/PCa,kernel,decision_boundary.py) -- PCA and decision_boundary methods with zircon TE chemistry data.
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
### Download from Github
```sh
git clone https://github.com/myscren/XXXXXX.git
cd XXXXXX
```
### Running the codes

The python codes require the Python complier installed on the PC or Laptop computer. To use the [related codes](#code-files), one must install the related  [Prerequisites](#prerequisites) on your Python complier.

* Update the path of [data files](#data-files) in python codes
* Run the python codes in [code files](#code-files)

Example:
```sh
python rf.py
```

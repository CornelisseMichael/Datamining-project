# Decision trees, Random forests and Gradient boosting for cancer type classification

This repository contains the Python code and Jupyter notebooks, of the Decision tree, Random forest and Gradient classifiers as described in the project report as part of the final <a href="https://github.com/Michael-Cornelisse/Datamining-project/blob/main/Problem_Research_Project_MichaelCornelisse_s1059020_NienkeHelmers_s1016904.pdf" target="_blank">project</a> of the course introduction to Datamining 2020 given at Radboud university.

In this project we have evaluated the performance of three different algorithms in a classification task on gene expression data to correctly classify the correct  cancer cell types acute myeloid leukaemia (AML) and acute lymphoblastic leukaemia (ALL) . Our initial hypotheses predicted that the Gradient boosting classifier would outperform the Random forest and Decision tree algorithms. This hypothesis turned out to be false with the Random forest being the best performing algorithm with a accuracy of 0.853, followed by Gradient Boost with an accuracy of  0.823 .The standard decision tree scores the lowest with an accuracy of 0.794



To run the code on your own machine make sure you have the following libraries installed:

***graphviz*** which can be installed by entering the following commands in anaconda terminal this should be the same for both mac, Linux and windows.

```
conda install graphviz
```

and

```
conda install python-graphviz
```

***xgboost*** library we developed this project on a windows machine to use this library in windows follow the following steps: 

1. Download the package from <a href="https://www.lfd.uci.edu/~gohlke/pythonlibs/#xgboost" target="_blank">this</a> website. `We used version: xgboost-1.3.1-cp38-cp38-win_amd64.whl`
2. Put the package in directory `C:\`
3. Open anaconda prompt
4. Type `cd C:\`
5. Type `pip install xgboost-1.3.1-cp38-cp38-win_amd64.whl` 
6. Type `conda update scikit-learn`

To install on Linux and mac refer to <a href="https://xgboost.readthedocs.io/en/latest/build.html" target="_blank">this</a> website.



The data comes from the [Gene expression dataset (Golub et al.)](https://www.kaggle.com/crawford/gene-expression.))[1].

[1] Gene expression dataset (Golub et al.). https://www.kaggle.com/crawford/gene-expression<br/>




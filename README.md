# pyrcc
A python implementation of Robust Continuous Clustering.

The original matlab implementation can be found [here](https://bitbucket.org/sohilas/robust-continuous-clustering).

Sklearn style demonstration:

![rcc_clustering](imgs/rcc_image_comparison.png?raw=true)

RCC is a clustering method introduced here: http://www.pnas.org/content/early/2017/08/28/1700770114

This is a port of the matlab implementation provided by the authors.

The code is self-contained in rcc.py

The following parameters are used in RCC:
- `k`: (int)(deafult `10`) number of neighbors used in the mutual KNN graph
- `verbose`: (bool)(default `True`) verbosity 
- `preprocessing`: (string)(default "none") one of 'scale', 'minmax', 'normalization', 'none'. How to preprocess the features
- `measure`: (string)(default "euclidean") one of 'cosine' or 'euclidean'. Paper used 'cosine'. Metric to use in constructing the mutual KNN graph
- `clustering_threshold`: (float)(default 1.0) controls how agressively to assign points to clusters.

A demonstration of how to use this is shown in demo.py, measuring the AMI (adjusted mutual information) using the pendigits dataset.

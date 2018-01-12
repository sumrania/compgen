# From SVM to LSTM: Classification of Time Series Gene Expression Data

To classify *Saccharomyces cerevisiae* time series gene expression profilesinto the 5 mitotic cell cycle phases - Early G1, Late G1, S, G2, M.

## Introduction

* Time series microarray experiments generate labelled temporal profiles which aresuitable for the following tasks: * Classification of unlabelled time course expression data. * Identification of dynamic biological networks.* Classification finds motivation most importantly in clinical applications whereprediction of a patient's response to a drug treatment is crucial for timely alterationof therapeutic strategies.* We implemented a novel CNN and LSTM, and compared the performance againsta temporal SVM, HMM and kNNDTW.

## Methods

1. 1-D Convolutional Neural Net:
 * `nn_code/cnn_timeseries.py`
![](/Users/sumrania/Documents/CMUSemesters/Spring 17/02-710 CompGenomics/Project/compgen/figures/cnn.png)

2. Long Short-Term Memory Network with ConvolutionalNetworks: 
 * `nn_code/lstm.py`
 
![](/Users/sumrania/Documents/CMUSemesters/Spring 17/02-710 CompGenomics/Project/compgen/figures/lstm.png)

3. K-Nearest Neighbors withDynamic Warping Distance: 
 * `DTW_kNN.py`

4. Gen-HMM and Disc-HMM: developed by Z.Bar-Joseph et al.: 
 * The code for this was downloaded from their paper and used as is.5. L1-norm Temporal SVM developed by Orsenigo et al.:
 * The code for this was downloaded from their paper and used as is.
## Results

![](/Users/sumrania/Documents/CMUSemesters/Spring 17/02-710 CompGenomics/Project/compgen/figures/results.png)

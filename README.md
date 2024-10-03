# HRAGNN-model
## HRAGNN: a cancer subtype identifying method using multi-omics data and heterogeneous graph neural network
With the development and progress of omics data, it has become an important topic for researchers to integrate multi-omics data to identify and classify cancer subtypes.   However, integrating multi-omics requires the use of relevant information among the omics data to screen out key features related to cancer subtypes, and how to efficiently integrate the omics data has become an important challenge. In this paper, we propose a new method, called HRAGNN for the identification of cancer subtypes based on multi-omics data. For cancer subtype identifying, integrating three multi-omics data: gene expression, miRNA expression, and DNA methylation data. HRAGNN consists of five steps: (i) Data preprocessing; (ii) Constructing similarity graph; (iii) Extracting features; (iv) Fusing features; (v) Features identifying;

### The Flow Chart:

![Flow Chart](https://github.com/1book1/HRAGNN-model/blob/main/Flow%20chart.png)

**The HRAGNN framework:**
- Data preprocessing; HRGNN first reduce data noise and standardize data to improve data quality;
- Constructing similarity graph; For each single omics data, we first construct the similarity graphs. Then we combine them to construct a final similarity graph; 
- Extracting features; We input the similarity graph into the relational attention network and then input the processed features into the Residual Graph Neural Network (RGNN), to obtain the features;
- Fusing features; We use Multi-View Fusion Network(MVFN) for features from multi-omics features;
- Identifying; Finally, we pass the obtained fusion features through a layer Softmax and get the class probability of each sample.

Experimental results show that this model has better results than other integrated multi-omics data classification methods.
# Requirements
- **Python 3.8**
- **PyTorch 2.3.1**
- **Tensorflow 2.4**
- **scikit-learn**
- **numpy**

# Dataset
Invasive breast cancer (BRCA) and glioblastoma multiforme (GBM) cancer samples, all datasets are publicly available on the TCGA website([https://www.cancer.gov](https://www.cancer.gov)). Each multiomics data is classified into four different subtypes. We also uploaded our preprocessed data into ``datas`` file.

# Running
You need to perform robustness processing on the data set and then divide the data set into training sets and test sets as input to the model, run the following commands:
```
Python Robustness_processing.py
```
```
Python normalize.py
```
## Trainning
Use the following command to get the training results:
```
Python train_test.py
```
# Result
The classification results are obtained after the end of the operation. You can evaluate results using four external performance metrics: Precision, Accuracy, Recall and F1 score, which proves the accuracy of classification.


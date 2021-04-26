# Machine Learning for Cyber Security cook book code

By Alexander Ospenko and Yasir Ali

# Chapter 1 Machine Learning for Cybersecurity

### Train test Splitting your data

![](img/ch1_train_test_split.png)

### Standardizing your data

![](img/ch1_standardizing_data.png)

### Principal Component Analysis

![](img/ch1_pca.png)

### Generating text using Markov Chains

Markov chains are simple stochastic models in which a system can exist in a number of
states. To know the probability distribution of where the system will be next, it suffices to
know where it currently is.

![](img/ch1_markov_chains.png)

### Performing Clustering algorithms

![](img/ch1_clustering.png)

### Training an XGBoost Classifier

![](img/ch1_XGBoost.png)

### Analysing time series using statsmodels

![](img/ch1_time_series.png)

### Anomaly detection with Isolation Forest

![](img/ch1_anomaly.png)

### Natural language processing using a hashing vectorizer and if-idf with scikit learn

A token is a unit of text. For example, we may specify that our tokens are words, sentences,
or characters. A count vectorizer takes textual input and then outputs a vector consisting of
the counts of the textual tokens. A hashing vectorizer is a variation on the count vectorizer
that sets out to be faster and more scalable, at the cost of interpretability and hashing
collisions. Though it can be useful, just having the counts of the words appearing in a
document corpus can be misleading. The reason is that, often, unimportant words, such as
the and a (known as stop words) have a high frequency of occurrence, and hence little
informative content. For reasons such as this, we often give words different weights to
offset this. The main technique for doing so is tf-idf, which stands for Term-Frequency,
Inverse-Document-Frequency. The main idea is that we account for the number of times a
term occurs, but discount it by the number of documents it occurs in

![](img/ch1_hash_vectorizer.png)

### Hyper parameter tuning with scikit optimize

![](img/ch1_hyperparameter_search.png)

Page 57






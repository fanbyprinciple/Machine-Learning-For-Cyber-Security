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

# Chapter 2 Machine Learning Based Malware Detection

### Malware Static Analysis

It can be name or yara.

1. comparing and finding out the hash value

![](ch2_hashes.png)

2. Looking at YARA rules

![](ch2_yara.png)

3. Looking at section headers

![](ch2_section_headers.png)

4. Looking at feature sections

![](ch2_feature_sections.png)

### Malware dynamic analysis

Unlike static analysis, dynamic analysis is a malware analysis technique in which the expert
executes the sample, and then studies the sample's behavior as it is being run. The main
advantage of dynamic analysis over static is that it allows you to bypass obfuscation by
simply observing how a sample behaves, rather than trying to decipher the sample's
contents and behavior.

Malware can be analysed by setting up a cuckoo sandbox

### using machine learning to detect the file type

to curate a dataset we will scrape github
Shows the function of pyGithub

Be careful with the github password while doing this exercise!

Page 73



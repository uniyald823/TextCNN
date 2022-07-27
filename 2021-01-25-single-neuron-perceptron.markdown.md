---
layout: post
read_time: true
show_date: true
title:  TextCNN
date:   
description: Text Classification using Convolutional Neural Networks
img: assets/img/posts/20210125/Perceptron.jpg 
tags: [machine learning, coding, neural networks]
author: Drishya Uniyal
github:  amaynez/Perceptron/
mathjax: yes
---
# Contents
1. Text Classification using CNN and other models
2. NNI for TextCNN
3. AutoKeras

## What is Text Classification?

Text Classification is one of the most valuable tasks of Natural language processing and is used to solve many problems. Instead of sorting out the categories manually from textual data, we can use text classification to solve this problem, which reduces time! 
#### Examples of Text Classification:
**Sentiment Analysis** "The food was excellent! - Positive"
**Language Detection** "The food was excellent! - English-en"
and more.

**Text Classification** is a process that provides labels to the set of texts or words in one, zero, or predefined label format. These labels tell us about the sentiment of the text.
<center><img src='pre.PNG' width="100" height="200"></center>
There may be different models through which we can train our machines to understand human languages. A machine can not directly understand the inputted data in the form of text; we need to transform this data and can do this in various ways:
1. Corpus in NLP: count the occurrence of each word in the sentence and provide it for the entire text.
2. CountVectorizer: make a vocabulary where every word has its special index number. 

Input:
    
    from sklearn.feature_extraction.text import CountVectorizer
    examplevectorizer = CountVectorizer()
    examplevectorizer.fit(example)
    examplevectorizer.vocabulary_ 

Output:


     {'The': 0,
      'food': 1,
      'was': 2,
      'excellent': 3,
    } 
    
This resulting vector is called a feature vector. Each word has its category in the feature vector, which is in numeric terms.
The data is prepared to fit in the model.There can be different models that can be used to fit the textual data, and in this article, Linear Regression, Naive Bayes, TextCNN, and RNN have been used to compare the models. 
As Yoon Kim gave his paper, the main focus remains on TextCNN and its variants.

##### Introduction to CNN 
Convolutional Neural Network is just a kind of neural network that performs well in image classification and computer vision. Its convolutional layer differs from other neural networks. We will be dealing with CNN for Text Classification.
Back in the 2014, [Yoon Kim](https://aclanthology.org/D14-1181) devised a very simple Convolutional Neural Network for Sentence Classification as a foundation for text classification and tried different variants of it to compare the performance. The main model focussed in this article will revolve around this architecture given by Yoon Kim.

<center><img src='download.png'></center>

The image above shows the CNN structure used by Yoon Kim, which is the basic CNN structure used for text classification. We consider text data as sequential data like data in time series, a one-dimensional matrix. We need to work with a one-dimensional convolution layer. The model idea is almost the same, but the data type and dimension of convolution layers changed. To work with TextCNN, we require a word embedding layer and a one-dimensional convolutional network. 

**What is Word Embedding?**
Word embedding represents the density of the word vector, which maps semantically similar words. It is a different way to preprocess the data. It does not consider the text as a human language but maps the structure of sets of words used in the corpus. They aim to map words into a geometric space which is called an embedding space.
The most common words do not have an extensive index in the embedding space.

One problem is the different lengths of words for which we need to specify the length of the word sequence and provide max length parameters to solve it. We need to use pad_sequence(), which pads the
sequence of words with zeros. Once the padding is done, we have to append zero value to matrices and now apply the deep learning model. This is how word embedding makes relations between words. In the next step, we will try to fit the TextCNN model.

We use a predefined word embedding available from the library for better performance. If the data is not embedded, then many embeddings are available open-source, like Glove and Word2Vec.
When we do dot product of vectors representing text, which turns zero if they belong to the same class, but if we do dot products of embedded words, we can find interrelation of words for a specific class. The kernel(filter layer) is passed over these embeddings to find convolutions, and the Max Pooling Layer of CNN dimensionally reduces these.
Lastly, the fully connected layers and the output activation function will give values for each class.
##### The Code:
**Dataset Used.**
The dataset used to test the models is Movie Review Dataset. (MR Dataset). In this dataset, the phrases are given and their corresponding sentiments.
<center><img src='data.PNG'></center>
**Yoon Kim code implementations and results:**

The architecture Kim uses is:
1. Sentences are represented as vectors of words.
2. These words are converted into (300D) vectors giving us a 2D representation for each sentence.

Approaches for creating these word vectors:
1. CNN-rand:(the basic model) where embeddings are randomly assigned to each word
2. CNN-static: word2vec is used to provide word embeddings. Unknown words are randomly initialized. These embeddings are kept fixed.
3. CNN-non-static: as above, but the vectors are fine-tuned (i.e., they can be changed) during training.
4. CNN-multichannel: Two sets of word vectors are used. Fine-tuning is done.
Convolutions are performed on these 2D representations with different window sizes (3, 4, and 5) are performed on the representations directly and then max pooled. Then the final predictions are made!

The accuracies for some of the datasets are are shown (as obtained by Yoon Kim)
|Model   |  MR | SST1|SST2 |
|:----:|:----:|:----:|:----:|
| CNN-rand |73.1 |45.0|82.7|
|CNN-static|81.0| 45.5|86.8|
|CNN-non-static|81.5|48.0|87.2|
|CNN-multichannel|81.1|47.4|88.1|

For the implementation of these, I took the Movies Review dataset (MR) and tried to obtain the results. MR datset is used to for all the models so that we can compare them.
My Results:
|Parameter   |  Value | 
|:----:|:----:|
 Kernel_Size | [3,4,5] |
 Dropout | 0.5 | 
 Optimizer | adam |
 Activation | Softmax |
 Glove Embedding | 42B.300D|
 Epochs | 10|
 
These values may change and hence the accuracies may change accordingly.
|Model   |  Accuracy | 
|:----:|:----:|
 CNN-rand | 37.27|
 CNN-static | 43.66 | 
 CNN-trainable | 50.05 | 
 CNN-binary-trainable | 49.48 |
  
 This was the basic implementation and results for Text Classification using CNN.

##### NNI — An AutoML Toolkit
AutoML is “Automatic Machine Learning,” a toolkit that runs machine learning models and implements experiments automatically.
In the context of the neural network, AutoML searches for different neural network architectures by taking into account the hyperparameters and training them to find the best fit model in a process called Neural Architecture Search.
Neural Network Intelligence (NNI) is a python AutoML package that works on Linux and Windows. It trains neural network models and finds a tuple of hyper-parameters that yields an optimal model.
The environment of NNI contains the main python file where the code is written and a config.yml file that connects the code. A .json file is where the search_space is defined, which contains different hyperparameters that the NNI framework uses and finds the best for the model. This can be defined inside the .yml file itself.
NNI runs experiments many times when it is training the model. Each of those attempts at applying a new configuration is called a Trial. NNI also provides a web interface that helps users investigate their trials and explore the experiment results.
A TextCNN model was implemented to test this framework. The dataset remained the same( MR Dataset), and then a simple CNN model was made, and different parameters were defined in the .yml file. After training, the web interface of NNI shows the result, and we can get a model with hyperparameters giving the best accuracy.

**Steps to run the code**
The folder contains the following files:
    
    main.py
    config.yml
    dataset
    
The folder contains the following files:
    
    redirect to the folder where the main.py and config.yml files are present.
    nnictl create  --config config.yml
    
The code runs on the localhost and we can check the values from there. An image of the web interface is shown below.

**Results**

##### AutoKeras
AutoKeras² is an open-source library that implements³ AutoML for deep learning using the Keras API. It automatically determines the best model and hyperparameters. 
The autokeras.TextClassifier class accepts the max_trials argument to set the maximum number of different Keras Models to try. We can specify the number of trails and epochs.
After the training process, we can use the best classifier to make predictions on the test set and evaluate performances. 
This was implemented for the MR dataset.
**Results**
|Model   |  Accuracy | 
|:----:|:----:|
|CNN|89|
AutoKeras describes the best model architecture and we can check this by summary().



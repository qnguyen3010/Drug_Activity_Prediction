#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  8 18:41:33 2017

@author: AaronNguyen
"""

import numpy as np
import pandas as pd
import scipy.sparse as sp
from numpy.linalg import norm
from collections import Counter, defaultdict
from scipy.sparse import csr_matrix
import scipy as sp
import random
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn import cross_validation
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import Perceptron
from sklearn.ensemble import RandomForestClassifier

# read the dataset
import csv   
def loadDataSet(filename, dataSet=[]):
	with open(filename, 'rt', ) as csvfile:
	    lines = csv.reader(csvfile, delimiter = '\t')
	    dataset = list(lines)
	    for x in range(len(dataset)):
	        dataSet.append(dataset[x])

trainSet=[]
testSet=[]
loadDataSet('train.dat.txt', trainSet)
loadDataSet('test.dat.txt', testSet)
print ('Dataset: ' + repr(len(trainSet)))
print ('Dataset: ' + repr(len(testSet)))

# seperate labels and compounds in the dataSet and testSet
pre_train = []
labels = []
pre_test = []

for i in range(len(trainSet)):
    pre_train.append(trainSet[i][1])
for j in range(len(trainSet)):
    labels.append(trainSet[j][0])
for k in range(len(testSet)):
    pre_test.append(testSet[k][0])
   
# seperate minor and major set of instances for under-sampling    
major_set = []
minor_set = []
for row in trainSet:
    if row[0] == '0':
        major_set.append(row)
    if row[0] == '1':
        minor_set.append(row)
        
print (len(major_set))
print (len(minor_set))

# we will pick a certain part of the major set 
numb = int(len(major_set)*0.3)
rand_major_set = random.sample(major_set,numb)

# new training dataset is created by combining a part of major set and whole minor set
new_trainingset = rand_major_set + minor_set
new_pre_train = []
new_labels = []
for m in range(len(new_trainingset)):
    new_pre_train.append(new_trainingset[m][1])
for n in range(len(new_trainingset)):
    new_labels.append(new_trainingset[n][0])
print (len(new_pre_train))
print (len(new_labels))

# extract attribute from both training and test dataset    
att_list =  set()
def attribute_selection(dataset):
    for rows in dataset:
        for attribute in  rows.split():
            att_list.add(attribute)
            
attribute_selection(new_pre_train) 
attribute_selection(pre_test)
print(len(att_list))

# use TfidVectorizer to create and normalize a sparse matrix
tf = TfidfVectorizer(norm='l2', vocabulary=list(att_list))

# create sparse matrix from both new training set and test set
test_matrix =  tf.fit_transform(pre_test)
new_train_matrix =  tf.fit_transform(new_pre_train)


# use SVD to reduce dimensions 
from sklearn.decomposition import TruncatedSVD
svd = TruncatedSVD(n_components=1000, n_iter=7, random_state=42)
svd_mod = svd.fit(new_train_matrix, new_labels)
new_train_reduced = svd_mod.transform(new_train_matrix)
test_reduced = svd_mod.transform(test_matrix)
print (new_train_reduced.shape)
print (test_reduced.shape)


# create knn classifier
from sklearn.neighbors import KNeighborsClassifier
clf = KNeighborsClassifier(n_neighbors = 25, metric = 'minkowski', p = 2)

# train our classifer using new training dataset and test set
X,y = new_train_reduced , new_labels

# create prediction based on the model
predict_list = clf.fit(X,y).predict(test_reduced)

# check the result
print(len(predict_list))
print(predict_list)

count1 = 0
count0 = 0
for number in predict_list:
    if number == '1':
        count1 += 1
    if number == '0':
        count0 += 1

print (count0)
print (count1)

# export the result into txt file
f = open("last.output.dat.txt", "w")
f.write("\n".join(map(lambda x: str(x), predict_list)))
f.close()















    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    



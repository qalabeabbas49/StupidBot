# importing libraries

import nltk
import os
from nltk.stem.lancaster import LancasterStemmer
import numpy as np
import tflearn
import tensorflow as tf
import random as rand
import json
import pickle

#loading Data

with open("/home/centos/qalab/noobbot/StupidBot/intents.json") as file:
    data = json.load(file)
    
# Text Preprocessing

#make lists from tags and patterns

#Create Bag of Words(BoW) vector containing collection of unique words


#iniializing empty lists

words = []  
labels = [] 
doc_x = []  
doc_y = []  

#looping through the data to fill the lists, patterns are converted to lower case and tokenized before adding to respective lists


for intent in data['intents']:
    for pattern in intent['patterns']:
        pattern = pattern.lower()
        
        #creating a list of words
        w = nltk.word_tokenize(pattern)
        words.extend(w)
        doc_x.append(w)
        doc_y.append(intent['tag']) 
        
        
    if intent['tag'] not in labels:
        labels.append(intent['tag'])
        


#Stemming (reduction of word to its root form) and Vectorization (converting sentences into to numerical form for ML/DL algorithms)

stemmer = LancasterStemmer()    
words = [stemmer.stem(w.lower()) for w in words if w not in "?"]    
words = sorted(list(set(words)))
labels = sorted(labels)

training = []
output = []

out_empty = [0 for _ in range(len(labels))]

for x, doc in enumerate(doc_x):
    bag = []
    wrds = [stemmer.stem(w) for w in doc]
    for w in words:
        if w in wrds:
            bag.append(1)
        else:
            bag.append(0)
    output_row = out_empty[:]
    output_row[labels.index(doc_y[x])] = 1
    training.append(bag)
    output.append(output_row)
    
#converting training data into NumPy arrays
training = np.array(training, dtype=object)
output = np.array(output)

#saving data to disk

with open("data.pickle", "wb") as f:
    pickle.dump((words, labels, training, output),f)



            
#Building a model
tf.compat.v1.reset_default_graph()

net = tflearn.input_data(shape= [None, len(training[0])])   
net = tflearn.fully_connected(net,8)
net = tflearn.fully_connected(net,8)    
net = tflearn.fully_connected(net, len(output[0]), activation = "softmax")  
net = tflearn.regression(net)

model = tflearn.DNN(net)


#fit the model on our data

model.fit(training, output, n_epoch=200, batch_size=8, show_metric=True)

model.save("model.tflearn")
            
            
            
            
            
        
        
        
        
        
        
        
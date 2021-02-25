
#################################
# Your name: eilon storzi
#################################

# Please import and use stuff only from the packages numpy, sklearn, matplotlib

import numpy as np
import matplotlib.pyplot as plt
import numpy.random
from sklearn.datasets import fetch_openml
import sklearn.preprocessing
import texttable as tt
"""
Assignment 3 question 1 skeleton.

Please use the provided function signature for the perceptron implementation.
Feel free to add functions and other code, and submit this file with the name perceptron.py
"""

def helper():
    mnist = fetch_openml('mnist_784')
    data = mnist['data']
    labels = mnist['target']

    neg, pos = "0", "8"
    train_idx = numpy.random.RandomState(0).permutation(np.where((labels[:60000] == neg) | (labels[:60000] == pos))[0])
    test_idx = numpy.random.RandomState(0).permutation(np.where((labels[60000:] == neg) | (labels[60000:] == pos))[0])

    train_data_unscaled = data[train_idx[:6000], :].astype(float)
    train_labels = (labels[train_idx[:6000]] == pos)*2-1

    validation_data_unscaled = data[train_idx[6000:], :].astype(float)
    validation_labels = (labels[train_idx[6000:]] == pos)*2-1

    test_data_unscaled = data[60000+test_idx, :].astype(float)
    test_labels = (labels[60000+test_idx] == pos)*2-1

    # Preprocessing
    train_data = sklearn.preprocessing.scale(train_data_unscaled, axis=0, with_std=False)
    validation_data = sklearn.preprocessing.scale(validation_data_unscaled, axis=0, with_std=False)
    test_data = sklearn.preprocessing.scale(test_data_unscaled, axis=0, with_std=False)
    return train_data, train_labels, validation_data, validation_labels, test_data, test_labels


def perceptron(data, labels):
    """
    returns: nd array of shape (data.shape[1],) or (data.shape[1],1) representing the perceptron classifier
    """
    w= [0 for i in range(784)]
    data=sklearn.preprocessing.normalize(data)
    for i in range(len(data)):
        y_predicted=np.dot(w,data[i])
        if(y_predicted>=0):
            y_predicted=1
        else:
            y_predicted=-1
        if(y_predicted!=labels[i]):
            xtyt=[data[i][j]*labels[i] for j in range(len(data[i]))]
            w = [w[i] + xtyt[i] for i in range (len(w))]
    return np.array(w)



#################################

# Place for additional code
def perceptron_acc(w,test_data,test_labels):
    cnt=0
    for i in range(len(test_data)):
        y_predicted = np.dot(w, test_data[i])
        if (y_predicted >= 0):
            y_predicted = 1
        else:
            y_predicted = -1
        if(y_predicted != test_labels[i]):
            cnt+=1
    acc = 1 - (cnt/len(test_data))
    return acc

#A
train_data, train_labels, validation_data, validation_labels, test_data, test_labels=helper()
table = tt.Texttable()
table.header(["n","average accuracy","5% accuracy percentile","95% accuracy percentile"])
avg_acc=[]
five_precent_acc=[]
ninty_five_precent_acc=[]
n_array=[5, 10, 50, 100, 500, 1000, 5000]
train_data_with_train_labels=np.column_stack((train_data,train_labels))
for n in n_array:
    acc_array=[]
    for j in range (100):
        np.random.shuffle(train_data_with_train_labels)
        data=[]
        labels=[]
        for i in range(n):
            data.append(train_data_with_train_labels[i][:-1])
            labels.append(train_data_with_train_labels[i][-1])
        w=perceptron(data,labels)
        acc_array.append(perceptron_acc(w,test_data,test_labels))
    acc_array.sort()
    avg_acc.append(sum(acc_array)/len(acc_array))
    five_precent_acc.append(acc_array[5])
    ninty_five_precent_acc.append(acc_array[95])
for row in zip(n_array,avg_acc,five_precent_acc,ninty_five_precent_acc):
    table.add_row(row)
s=table.draw()
#print(s)
#B
w = perceptron(train_data, train_labels)
plt.imshow(np.reshape(w, (28, 28)), interpolation='nearest')
#plt.show()
#C
w = perceptron(train_data, train_labels)
print("the accuracy of perceptron algorithm is:" , perceptron_acc(w,test_data,test_labels))

#D
w = perceptron(train_data, train_labels)
for i in range(len(test_data)):
    y_predicted = np.dot(w, test_data[i])
    if (y_predicted >= 0):
        y_predicted = 1
    else:
        y_predicted = -1
    if (y_predicted != test_labels[i]):
         print("the picture has labeled" ,test_labels[i])
         plt.imshow(np.reshape(test_data[i], (28, 28)), interpolation='nearest')
         plt.show()
         break
#################################

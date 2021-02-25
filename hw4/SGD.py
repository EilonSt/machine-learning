
#################################
# Your name:Eilon Storzi
#################################

# Please import and use stuff only from the packages numpy, sklearn, matplotlib

import numpy as np
import numpy.random
from sklearn.datasets import fetch_openml
import sklearn.preprocessing
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
"""
Assignment 3 question 2 skeleton.

Please use the provided function signature for the SGD implementation.
Feel free to add functions and other code, and submit this file with the name sgd.py
"""


def helper_hinge():
    mnist = fetch_openml('mnist_784')
    data = mnist['data']
    labels = mnist['target']

    neg, pos = "0", "8"
    train_idx = numpy.random.RandomState(0).permutation(np.where((labels[:60000] == neg) | (labels[:60000] == pos))[0])
    test_idx = numpy.random.RandomState(0).permutation(np.where((labels[60000:] == neg) | (labels[60000:] == pos))[0])

    train_data_unscaled = data[train_idx[:6000], :].astype(float)
    train_labels = (labels[train_idx[:6000]] == pos) * 2 - 1

    validation_data_unscaled = data[train_idx[6000:], :].astype(float)
    validation_labels = (labels[train_idx[6000:]] == pos) * 2 - 1

    test_data_unscaled = data[60000 + test_idx, :].astype(float)
    test_labels = (labels[60000 + test_idx] == pos) * 2 - 1

    # Preprocessing
    train_data = sklearn.preprocessing.scale(train_data_unscaled, axis=0, with_std=False)
    validation_data = sklearn.preprocessing.scale(validation_data_unscaled, axis=0, with_std=False)
    test_data = sklearn.preprocessing.scale(test_data_unscaled, axis=0, with_std=False)
    return train_data, train_labels, validation_data, validation_labels, test_data, test_labels


def helper_ce():
    mnist = fetch_openml('mnist_784')
    data = mnist['data']
    labels = mnist['target']

    train_idx = numpy.random.RandomState(0).permutation(np.where((labels[:8000] != 'a'))[0])
    test_idx = numpy.random.RandomState(0).permutation(np.where((labels[8000:10000] != 'a'))[0])

    train_data_unscaled = data[train_idx[:6000], :].astype(float)
    train_labels = labels[train_idx[:6000]]

    validation_data_unscaled = data[train_idx[6000:8000], :].astype(float)
    validation_labels = labels[train_idx[6000:8000]]

    test_data_unscaled = data[8000 + test_idx, :].astype(float)
    test_labels = labels[8000 + test_idx]

    # Preprocessing
    train_data = sklearn.preprocessing.scale(train_data_unscaled, axis=0, with_std=False)
    validation_data = sklearn.preprocessing.scale(validation_data_unscaled, axis=0, with_std=False)
    test_data = sklearn.preprocessing.scale(test_data_unscaled, axis=0, with_std=False)
    return train_data, train_labels, validation_data, validation_labels, test_data, test_labels


def SGD_hinge(data, labels, C, eta_0, T):
    """
    Implements Hinge loss using SGD.
    """
    w = [0 for i in range(784)]
    for i in range(1, T + 1):
        eta_T = eta_0 / i
        rand_i = np.random.randint(0, len(data))
        if (np.dot(w, data[rand_i]) * labels[rand_i] < 1):
            w = np.dot((1 - eta_T), w)
            tmp = np.dot(data[rand_i],(eta_T * C * labels[rand_i]))
            w = [w[j] + tmp[j] for j in range(len(w))]
        else:
            w = np.dot((1 - eta_T), w)
    return w



def SGD_ce(data, labels, eta_0, T):
    """
    Implements multi-class cross entropy loss using SGD.
    """
    w = [0 for i in range(784)]
    classifiers = [w for k in range(10)]
    for i in range(1, T + 1):
        rand_i = np.random.randint(0, len(data))
        grad = grad_ce(data[rand_i], labels[rand_i], classifiers)
        grad= [np.dot(grad[j],data[rand_i]) for j in range(len(grad))]
        for j in range(len(grad)):
            grad[j]=np.dot(eta_0,grad[j])
        for j in range(len(classifiers)):
            classifiers[j]=[classifiers[j][k] - grad[j][k] for k in range(len(classifiers[j]))]
    return classifiers

#################################

# Place for additional code
def hinge_loss_acc(w,validation_data,validation_labels):
    cnt=0
    for i in range(len(validation_data)):
        y_predicted = np.dot(w, validation_data[i])
        if (y_predicted >= 0):
            y_predicted = 1
        else:
            y_predicted = -1
        if(y_predicted != validation_labels[i]):
            cnt+=1
    acc = 1 - (cnt/len(validation_data))
    return acc
def softmax(z):
    e = np.exp(z-np.max(z))
    s = np.sum(e)
    return e/s

def grad_ce(data,label,classifiers):
    g=[np.dot(classifiers[i],data) for i in range(len(classifiers))]
    g=softmax(g)
    for i in range(len(classifiers)):
        if (i==int(label)):
            g[i]-=1
    return g

def predict (data,classifiers):
    index=0
    p = [np.dot(classifiers[j], data) for j in range(len(classifiers))]
    p=softmax(p)
    max_pred=np.max(p)
    for j in range(len(classifiers)):
        if(p[j]==max_pred):
            index=j
            break
    return index

def cross_entropy_loss_acc(classifiers,validation_data,validation_labels):
    cnt=0
    for i in range(len(validation_data)):
        y_predicted = predict(validation_data[i],classifiers)
        if(y_predicted != int (validation_labels[i])):
            cnt+=1
    acc = 1 - (cnt/len(validation_data))
    return acc

#A

train_data, train_labels, validation_data, validation_labels, test_data, test_labels=helper_hinge()
T=1000
C=1
eta_0=10**-5
etas=[]
acc_avg=[]
step=2*10**-5
for i in range(10):
    acc_list = []
    for k in range(5):
        if(k==1):
            eta_0=step
        for j in range(10):
            w=SGD_hinge(train_data,train_labels,C,eta_0,T)
            acc_list.append(hinge_loss_acc(w,validation_data,validation_labels))
        acc_avg.append(np.average(acc_list))
        etas.append(eta_0)
        eta_0=eta_0+step
    step=step*10
max_acc=0
index=0
for i in range(len(acc_avg)):
    if(acc_avg[i]>max_acc):
        index=i
        max_acc=acc_avg[i]
#print(etas[index])
#print(max_acc)
blue_patch = mpatches.Patch(color='blue', label='accuracy as function of eta_0')
plt.legend(handles=[blue_patch])
plt.xscale('log')
plt.xlabel("eta_0")
plt.ylabel("avg accuracy")
plt.plot(etas,acc_avg)
#plt.show()

#B
train_data, train_labels, validation_data, validation_labels, test_data, test_labels=helper_hinge()
T=1000
eta_0=etas[index]
C=10**-5
C_s=[]
acc_avg=[]
step=2*10**-5
for i in range(10):
    acc_list = []
    for k in range(5):
        if(k==1):
            C=step
        for j in range(10):
            w=SGD_hinge(train_data,train_labels,C,eta_0,T)
            acc_list.append(hinge_loss_acc(w,validation_data,validation_labels))
        acc_avg.append(np.average(acc_list))
        C_s.append(C)
        C=C+step
    step=step*10
max_acc=0
index=0
for i in range(len(acc_avg)):
    if(acc_avg[i]>max_acc):
        index=i
        max_acc=acc_avg[i]
#print(C_s[index])
#print(max_acc)
blue_patch = mpatches.Patch(color='blue', label='accuracy as function of C')
plt.legend(handles=[blue_patch])
plt.xscale('log')
plt.xlabel("C")
plt.ylabel("avg accuracy")
plt.plot(C_s,acc_avg)
#plt.show()

#C+D
train_data, train_labels, validation_data, validation_labels, test_data, test_labels=helper_hinge()
T=20000
C=C_s[index]
w=SGD_hinge(train_data,train_labels,C,eta_0,T)
plt.imshow(np.reshape(w, (28, 28)), interpolation='nearest')
#plt.show()
acc=hinge_loss_acc(w,test_data,test_labels)
print("max accuracy=",acc)

#A
train_data, train_labels, validation_data, validation_labels, test_data, test_labels=helper_ce()
T=1000
eta_0=10**-5
etas=[]
acc_avg=[]
for i in range(10):
    acc_list = []
    for j in range(10):
        w=SGD_ce(train_data,train_labels,eta_0,T)
        acc_list.append(cross_entropy_loss_acc(w,validation_data,validation_labels))
        #print(acc_list[j],"run:",j)
    acc_avg.append(np.average(acc_list))
    #print(acc_avg[i], "avg accuracy")
    etas.append(eta_0)
    eta_0=eta_0*10
max_acc=0
index=0
for i in range(len(acc_avg)):
    if(acc_avg[i]>max_acc):
        index=i
        max_acc=acc_avg[i]
blue_patch = mpatches.Patch(color='blue', label='accuracy as function of eta_0')
plt.legend(handles=[blue_patch])
plt.xscale('log')
plt.xlabel("eta_0")
plt.ylabel("avg accuracy")
plt.plot(etas,acc_avg)
#plt.show()

#B+C
train_data, train_labels, validation_data, validation_labels, test_data, test_labels=helper_ce()
T=20000
eta_0=etas[index]
w=SGD_ce(train_data,train_labels,eta_0,T)
for i in range(len(w)):
    plt.imshow(np.reshape(w[i], (28, 28)), interpolation='nearest')
    #plt.show()
acc=cross_entropy_loss_acc(w,test_data,test_labels)
print("max accuracy=",acc)

#################################

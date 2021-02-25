import numpy.random
import random
from sklearn.datasets import fetch_openml
import matplotlib.pyplot as plt
mnist = fetch_openml('mnist_784')

data = mnist['data']

labels = mnist['target']


idx = numpy.random.RandomState(0).choice(70000, 11000)
train = data[idx[:10000], :].astype(int)
train_labels = labels[idx[:10000]]
test = data[idx[10000:], :].astype(int)
test_labels = labels[idx[10000:]]
#A
def nearest_neighbor(data, labels, query, k):
    nearest_neighbors_list = [0]*k
    nnlabel_list = [0]*k
    count_list=[0]*10
    for i in range(len(data)):
        x=(((query-data[i])**2).sum())**0.5
        if(i<k):
            nearest_neighbors_list[i]=x
            nnlabel_list[i]=labels[i]
        else:
            val = max(nearest_neighbors_list)
            if(x < val):
                m=nearest_neighbors_list.index(val)
                nearest_neighbors_list[m] = x
                nnlabel_list[m] = labels[i]

    for i in range(k):
        count_list[int(nnlabel_list[i])]+=1
    label=[]
    maximum=count_list[0]
    for i in range(10):
        if(count_list[i]>maximum):
            maximum=count_list[i]
            label=[]
            label.append(i)
        elif(count_list[i]==maximum):
            label.append(i)
        else:
            continue

    if(len(label)>1):
        tmp=random.randint(0,len(label)-1)
        query_label=label[tmp]
    else:
        query_label=label[0]
    return query_label
#B
sum=0
for i in range(len(test)):
    z=nearest_neighbor(train[:1001],train_labels[:1001],test[i],10)
    if(z!=int(test_labels[i])):
            sum += 1
print("the prediction accuracy is:",(1-(sum/len(test)))*100)

#C
result=[0]*100
for j in range(1,101):
    sum=0
    for i in range(len(test)):
        z=nearest_neighbor(train[:1001],train_labels[:1001],test[i],j)
        if(z!=int(test_labels[i])):
                sum += 1
    result[j-1]=((1-(sum/len(test)))*100)

plt.plot([j for j in range(1,101)],result)
plt.show()
#D
res=[0]*50
for i in range(100,5001,100):
    sum=0
    for j in range(len(test)):
        z = nearest_neighbor(train[:i+1], train_labels[:i+1], test[j], 1)
        if(z!=int(test_labels[j])):
            sum += 1
        res[((i//100)-1)]=((1-(sum/len(test)))*100)
plt.plot([i for i in range(100, 5001, 100)], res)
plt.show()

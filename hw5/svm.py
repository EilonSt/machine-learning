#################################
# Your name: Eilon Storzi
#################################

# Please import and use stuff only from the packages numpy, sklearn, matplotlib
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.datasets import make_blobs
import matplotlib.patches as mpatches
"""
Q4.1 skeleton.

Please use the provided functions signature for the SVM implementation.
Feel free to add functions and other code, and submit this file with the name svm.py
"""

# generate points in 2D
# return training_data, training_labels, validation_data, validation_labels
def get_points():
    X, y = make_blobs(n_samples=120, centers=2, random_state=0, cluster_std=0.88)
    return X[:80], y[:80], X[80:], y[80:]


def create_plot(X, y, clf):
    plt.clf()

    # plot the data points
    plt.scatter(X[:, 0], X[:, 1], c=y, s=30, cmap=plt.cm.PiYG)

    # plot the decision function
    ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    # create grid to evaluate model
    xx = np.linspace(xlim[0] - 2, xlim[1] + 2, 30)
    yy = np.linspace(ylim[0] - 2, ylim[1] + 2, 30)
    YY, XX = np.meshgrid(yy, xx)
    xy = np.vstack([XX.ravel(), YY.ravel()]).T
    Z = clf.decision_function(xy).reshape(XX.shape)

    # plot decision boundary and margins
    ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.5,
               linestyles=['--', '-', '--'])


def train_three_kernels(X_train, y_train, X_val, y_val):
    """
    Returns: np.ndarray of shape (3,2) :
                A two dimensional array of size 3 that contains the number of support vectors for each class(2) in the three kernels.
    """
    C=1000
    linear_svm=svm.SVC(C,kernel="linear")
    linear_svm.fit(X_train,y_train)
    LSV=linear_svm._n_support
    create_plot(X_train,y_train,linear_svm)
    plt.title("number of SV="+ str(sum(LSV)))
    plt.suptitle("Linear SVM model")
    #plt.show()
    quadratic_svm=svm.SVC(C,kernel="poly",degree=2)
    quadratic_svm.fit(X_train,y_train)
    QSV=quadratic_svm.n_support_
    create_plot(X_train,y_train,quadratic_svm)
    plt.title("number of SV="+ str(sum(QSV)))
    plt.suptitle("Quadratic SVM model")
    #plt.show()
    rbf_svm = svm.SVC(C)
    rbf_svm.fit(X_train, y_train)
    RSV = rbf_svm.n_support_
    create_plot(X_train, y_train, rbf_svm)
    plt.title("number of SV=" + str(sum(RSV)))
    plt.suptitle("Rbf SVM model")
    #plt.show()
    return np.array([LSV,QSV,RSV])



def linear_accuracy_per_C(X_train, y_train, X_val, y_val):
    """
        Returns: np.ndarray of shape (11,) :
                    An array that contains the accuracy of the resulting model on the VALIDATION set.
    """
    C=10**-5
    C_list=[]
    Vacc_list=[]
    Tacc_list=[]
    for i in range(10):
        linear_svm = svm.SVC(C, kernel="linear")
        linear_svm.fit(X_train, y_train)
        create_plot(X_train, y_train,linear_svm)
        plt.title("with C=" + str(C))
        plt.suptitle("Linear SVM model")
        plt.show()
        cnt=0
        cnt2=0
        Vpredictions = linear_svm.predict(X_val)
        Tpredictions = linear_svm.predict(X_train)
        for j in range (len(Vpredictions)):
            if(Vpredictions[j]==y_val[j]):
                cnt+=1
        for j in range (len(Tpredictions)):
            if (Tpredictions[j] == y_train[j]):
                cnt2 += 1
        Vacc_list.append(cnt/len(X_val))
        Tacc_list.append(cnt2 / len(X_train))
        C_list.append(C)
        C=C*10
    Tindex=0
    Vindex=0
    for i in range(len(C_list)):
        if(Tacc_list[i]>Tacc_list[Tindex]):
            Tindex=i
        if (Vacc_list[i] > Vacc_list[Vindex]):
            Vindex = i
    print("the best C for the training data is:",C_list[Tindex])
    print("the best C for the validation data is:",C_list[Vindex])
    plt.xscale('log')
    plt.xlabel("C")
    plt.ylabel("accuracy")
    plt.title("accuracy as function of C")
    plt.plot(C_list, Vacc_list,marker="*",label="accuracy on the validation data as function of C")
    plt.plot(C_list,Tacc_list,marker="o",label="accuracy on the training data as function of C")
    plt.legend(bbox_to_anchor=(0.3, 0.5), loc='best')
    #plt.show()
    return np.array(Vacc_list)

def rbf_accuracy_per_gamma(X_train, y_train, X_val, y_val):
    """
        Returns: np.ndarray of shape (11,) :
                    An array that contains the accuracy of the resulting model on the VALIDATION set.
    """
    gamas=[10**i for i in range(-5,6)]
    Vacc_list=[]
    Tacc_list=[]
    for g in gamas:
        rbf_svm = svm.SVC(C=10,kernel='rbf',gamma=g)
        rbf_svm.fit(X_train, y_train)
        create_plot(X_train, y_train, rbf_svm)
        plt.title("with gama=" + str(g))
        plt.suptitle("Rbf SVM model")
        #plt.show()
        cnt = 0
        cnt2 = 0
        Vpredictions = rbf_svm.predict(X_val)
        Tpredictions = rbf_svm.predict(X_train)
        for j in range(len(Vpredictions)):
            if (Vpredictions[j] == y_val[j]):
                cnt += 1
        for j in range(len(Tpredictions)):
            if (Tpredictions[j] == y_train[j]):
                cnt2 += 1
        Vacc_list.append(cnt / len(X_val))
        Tacc_list.append(cnt2 / len(X_train))
    Tindex=0
    Vindex=0
    for i in range(len(gamas)):
        if(Tacc_list[i]>Tacc_list[Tindex]):
            Tindex=i
        if (Vacc_list[i] > Vacc_list[Vindex]):
            Vindex = i
    print("the best gama for the training data is:",gamas[Tindex])
    print("the best gama for the validation data is:",gamas[Vindex])
    plt.xscale('log')
    plt.xlabel("gama")
    plt.ylabel("accuracy")
    plt.title("accuracy as function of gama")
    plt.plot(gamas, Vacc_list,marker="*",label="accuracy on the validation data as function of gama")
    plt.plot(gamas,Tacc_list,marker="o",label="accuracy on the training data as function of gama")
    plt.legend(bbox_to_anchor=(0.4, 0.05), loc='lower center')
    #plt.show()


X_train, y_train, X_val, y_val=get_points()
train_three_kernels(X_train, y_train, X_val, y_val)
linear_accuracy_per_C(X_train, y_train, X_val, y_val)
rbf_accuracy_per_gamma(X_train, y_train, X_val, y_val)

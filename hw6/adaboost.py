
#################################
# Your name: Eilon Storzi
#################################

# Please import and use stuff only from the packages numpy, sklearn, matplotlib.
from matplotlib import pyplot as plt
import numpy as np
from process_data import parse_data
import matplotlib.patches as mpatches


np.random.seed(7)


def run_adaboost(X_train, y_train, T):
    """
    Returns:

        hypotheses :
            A list of T tuples describing the hypotheses chosen by the algorithm.
            Each tuple has 3 elements (h_pred, h_index, h_theta), where h_pred is
            the returned value (+1 or -1) if the count at index h_index is <= h_theta.

        alpha_vals :
            A list of T float values, which are the alpha values obtained in every
            iteration of the algorithm.
    """
    # TODO: add your code here
    hypotheses=[]
    alpha_vals=[]
    num_samples,num_features=X_train.shape
    Dt=np.full(num_samples, (1 / num_samples))
    for i in range(1,T+1):
        print(i)
        min_error= float('inf')
        for j in range(num_features):
            X_col=X_train[:,j]
            thetas=np.unique(X_col)
            thresholds=np.sort(thetas)
            new_thresholds = np.zeros(len(thresholds) + 1)
            new_thresholds[0] = thresholds[0] - 1
            new_thresholds[len(thresholds)] = thresholds[len(thresholds) - 1] + 1
            for k in range(1, len(thresholds)):
                new_thresholds[k] = (thresholds[k - 1] + thresholds[k]) / 2
            for theta in new_thresholds:
                hypo=-1
                predctions = np.ones(num_samples)
                predctions[X_col < theta] = -1
                missclassified=Dt[y_train!=predctions]
                error=np.sum(missclassified)
                if (error > 0.5):
                    error = 1 - error
                    hypo = 1

                if (error < min_error):
                    min_error = error
                    hypo_type=hypo
                    best_theta=theta
                    best_feature=j
        alpha=0.5*np.log(((1.0 - min_error+1e-10)/(min_error+1e-10)))
        alpha_vals.append(alpha)
        hypotheses.append((hypo_type,best_feature,best_theta))
        predctions=predict(X_train,hypotheses[-1])
        Dt= Dt * np.exp(-alpha*y_train*predctions)
        Dt= Dt/np.sum(Dt)
    return hypotheses,alpha_vals





##############################################
# You can add more methods here, if needed.
def predict(X_train,hypoteses):
    num_samples=X_train.shape[0]
    X_col=X_train[:,hypoteses[1]]
    predctions=np.ones(num_samples)
    if(hypoteses[0]==-1):
        predctions[X_col<hypoteses[2]]=-1
    else:
        predctions[X_col > hypoteses[2]] = -1
    return predctions


def ada_boost_pred(hypotheses,alpha_vals,X):
    y_predict =[alpha_vals[j]*predict(X,hypotheses[j]) for j in range(len(alpha_vals))]
    predictions=np.sum(y_predict,axis=0)
    predictions=np.sign(predictions)
    return predictions

def error(hypotheses,alpha_vals,X,Y):
    y_pred=ada_boost_pred(hypotheses,alpha_vals,X)
    acc=(np.sum(Y==y_pred)) /len(Y)
    return 1-acc

def exp_error(hypotheses,alpha_vals,X,Y):

    y_predict = [alpha_vals[j] * predict(X, hypotheses[j]) for j in range(len(alpha_vals))]
    predictions = np.sum(y_predict, axis=0)
    pred=[-Y[i] * predictions[i] for i in range(len(Y))]
    return np.sum(np.exp(pred))/len(Y)




##############################################


def main():
    data = parse_data()
    if not data:
        return
    (X_train, y_train, X_test, y_test, vocab) = data
    #A

    T=80
    hypotheses, alpha_vals = run_adaboost(X_train, y_train, T)
    train_error=[]
    test_error=[]
    for i in range(T):
        train_error.append(error(hypotheses[:i+1],alpha_vals[:i+1],X_train,y_train))
        test_error.append(error(hypotheses[:i+1],alpha_vals[:i+1],X_test,y_test))
    plt.plot([i for i in range(T)], train_error, color='blue', marker='o')
    plt.plot([i for i in range(T)], test_error, color='red', marker='+')
    plt.xlabel("t - iteration")
    plt.ylabel("error")
    blue_patch=mpatches.Patch(color='blue',label='train error')
    red_patch=mpatches.Patch(color='red',label='test error')
    plt.legend(handles=[blue_patch,red_patch])
    #plt.show()
 
    #B
    T = 10
    hypotheses, alpha_vals = run_adaboost(X_train, y_train, T)
    for i in range(T):
        print("the hypotheses is: ",hypotheses[i])
        print("the word is:",vocab[hypotheses[i][1]])
        print("and her wight is:", alpha_vals[i])

    #C
    T=80
    hypotheses, alpha_vals = run_adaboost(X_train, y_train, T)
    train_error=[]
    test_error=[]
    for i in range(T):
        print(i)
        train_error.append(exp_error(hypotheses[:i+1],alpha_vals[:i+1],X_train,y_train))
        test_error.append(exp_error(hypotheses[:i+1],alpha_vals[:i+1],X_test,y_test))
    plt.plot([i for i in range(T)], train_error, color='blue', marker='o')
    plt.plot([i for i in range(T)], test_error, color='red', marker='+')
    plt.xlabel("t - iteration")
    plt.ylabel("exp error")
    blue_patch=mpatches.Patch(color='blue',label='train error')
    red_patch=mpatches.Patch(color='red',label='test error')
    plt.legend(handles=[blue_patch,red_patch])
    #plt.show()
    ##############################################
    # You can add more methods here, if needed.



    ##############################################

if __name__ == '__main__':
    main()

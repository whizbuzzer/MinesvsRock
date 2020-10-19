#######################################################################################################################
#######################################################################################################################

################################
# Project 2: Machine Learning  #
#            Mine vs Rock      #
# Created by Aniket N Prabhu   #
################################

#######################################################################################################################

import numpy as np  # needed for arrays
import pandas as pd  # For reading the csv file
import matplotlib.pyplot as plt  # For plotting accuracy vs number of components
from sklearn.model_selection import train_test_split  # For splitting data
from sklearn.preprocessing import StandardScaler  # For scaling data
from sklearn.decomposition import PCA  # For using PCA
from sklearn.metrics import accuracy_score, confusion_matrix  # For calculating accuracy and the confusion matrix
from warnings import filterwarnings  # For ignoring redundant warnings
from sklearn.neural_network import MLPClassifier  # For making the neural network using the Multi Layer Perceptron

#######################################################################################################################

datson = pd.read_csv('sonar_all_data_2.csv', header=None)
X = datson.iloc[:, 0:60].values  # Features
Y = datson.iloc[:, 60].values  # Classification
# print(X)
# print(Y)

# Splitting data 70% for training and 30% for testing #
Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=0.3, random_state=0)
# print(Ytest)
SS = StandardScaler()  # Standardize features by removing the mean and scaling to unit variance
SS.fit(Xtrain)  # Fit calculates how to change the data based on what you're doing
stdXtrain = SS.transform(Xtrain)
stdXtest = SS.transform(Xtest)  # Mean and Std Dev not required for test sets

#######################################################################################################################

# Using Principal Component Analysis (PCA) #
accuracy = []  # To store the accuracy values
confmat = []  # To store the confusion matrices

# For loop made to determine what number of components will give the maximum accuracy #
for i in range(1, 61):
    pca = PCA(n_components=i)
    pcatrainX = pca.fit_transform(stdXtrain)
    pcatestX = pca.transform(stdXtest)
    # Using Multi-Layer Perceptron (MLP) to create neural network #
    mlp = MLPClassifier(hidden_layer_sizes=100, activation='logistic', max_iter=2000, alpha=0.00001,
                        solver='lbfgs', tol=0.0001, random_state=0)
    filterwarnings('ignore')  # For ignoring convergence warnings
    mlp.fit(pcatrainX, Ytrain)
    PredY = mlp.predict(pcatestX)
    acc1 = accuracy_score(Ytest, PredY)
    accuracy.append(acc1)
    print("\nNumber of components: ", i)
    print("\nAccuracy achieved: ", acc1)
    cmat = confusion_matrix(Ytest, PredY)
    confmat.append(cmat)
    # print("\nConfusion matrix:\n", cmat)  # For debugging

# print(confmat)                 # For debugging

# Finding maximum accuracy and the number of components corresponding to it #
maxAcc = max(accuracy)           # Maximum value from the 'accuracy' array
indMax = accuracy.index(maxAcc)  # For finding number of components corresponding to maximum accuracy
# indMax = int(indMax)
maxComp = indMax + 1
# MaxYtest = Ytest[indMax]
# MaxYPred = YPred[indMax]

print('\nMaximum accuracy: %.2f' % maxAcc)
print('\nNumber of components for maximum accuracy: ', maxComp)

# Confusion matrix corresponding to maximum accuracy #
cmat1 = confmat[indMax]
print("\nConfusion matrix for maximum accuracy:\n", cmat1)

# Number of components vs Accuracy achieved plot #
plt.plot(np.arange(1, 61), accuracy)  # Setting the range so that the number of components do not start from 0.
plt.title('Number of components vs Accuracy achieved')
plt.xlabel('Number of components')
plt.ylabel('Accuracy achieved')
plt.grid()
plt.show()

#######################################################################################################################
#######################################################################################################################

###################### Aiyappa CODE for yeast_cell_cycle_phase
import os
import numpy as np
import dtw_1nn as dtw
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold,LeaveOneOut
from sklearn.metrics import confusion_matrix

np.random.seed(100)
######################### Import the data
X_train = np.zeros((377, 17))
Y_train = []
with open("train_data.txt") as dataset :
    count = 0
    for line in dataset:
            string_data = line.split('\t')
            # print string_data
            Y_train.append(int(string_data[0]))
            # print len(string_data)
            expression_values = map(float,string_data[3:20])
            X_train[count, :] = expression_values
            count += 1

Y_train = np.array(Y_train)

########################## merge and shuffle the dataset
Y_train = Y_train[:,np.newaxis]
joint = np.hstack((X_train,Y_train))
# np.random.shuffle(joint)
Y_train = joint[:,17]
X_train = joint[:,0:17]

####################### split the data into training and validation sets
X_run = X_train[0:350,:]
Y_run = Y_train[0:350]
X_validate = X_train[350:377,:]
Y_validate = Y_train[350:377]

######################### create model, populate with X_run and Y_run, predict labels with probabilities for X_validate and Y_validate
print "\nMODE: USER INPUT VALUE OF K. Do not set too large!!\n"
num_neighbours = input("Enter value of k!\n")
model = dtw.KnnDtw(n_neighbors=num_neighbours,max_warping_window=2)
model.fit(X_run,Y_run)
preds,probs = model.predict(X_validate)
print "\nThe accuracy of DTW-",num_neighbours,"NN classifier is",1.0*np.sum(preds==Y_validate)/len(Y_validate),"(For a fixed holdout set)."


########################TESTS FOR OPTIMIZATION#####################################################################

######################### check for the optimal K
print "\nFINDING OPTIMAL k-VALUE(For a fixed holdout set).\n"
accuracy = []
ks = []
for k in range(1,30):
    ks.append(k)
    model = dtw.KnnDtw(n_neighbors=k, max_warping_window=2)
    model.fit(X_run, Y_run)
    preds, probs = model.predict(X_validate)
    accuracy.append(1.0 * np.sum(preds == Y_validate) / len(Y_validate))
    # print "\n", 1.0 * np.sum(preds == Y_validate) / len(Y_validate)

plt.plot(ks,accuracy)
plt.show()


############################## cross validation using LeaveOneOut
print "\nLEAVE-ONE-OUT CROSS VALIDATION.\n"
indices = range(0,len(Y_train))
loo = LeaveOneOut()
accuracies = []
for train_index, test_index in loo.split(indices):

   X_train_l, X_test= X_train[train_index], X_train[test_index]
   Y_train_l, Y_test = Y_train[train_index], Y_train[test_index]

   model = dtw.KnnDtw(n_neighbors=1,max_warping_window=2)
   model.fit(X_train_l, Y_train_l)
   preds,_= model.predict(X_test)
   accuracies.append(1.0*np.sum(preds==Y_test)/len(Y_test))

print "\nLeave-One_Out Cross Validation accuracy: ",1.0*sum(accuracies)/len(Y_train)



########################## cross validation using KFold

for num_neigh in range(1,21):
    num_splits = 20                     #fix value of k
    print "\n",num_splits,"-FOLD CROSS VALIDATAION. This will take some time!!\n"
    indices = range(0,len(Y_train))
    kf = KFold(n_splits=num_splits)
    accuracies = []
    for train_index, test_index in kf.split(indices):

       X_train_l, X_test= X_train[train_index], X_train[test_index]
       Y_train_l, Y_test = Y_train[train_index], Y_train[test_index]

       model = dtw.KnnDtw(n_neighbors=num_neigh,max_warping_window=2)
       model.fit(X_train_l, Y_train_l)
       preds,_= model.predict(X_test)
       accuracies.append(1.0*np.sum(preds==Y_test)/len(Y_test))

    print "\nFor number of neighbours: ",num_neigh,", the  accuracy is ",1.0*sum(accuracies)/num_splits


print"\n RUN COMPLETE!!"
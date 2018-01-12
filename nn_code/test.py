import numpy as np
import os, sys
import keras
from keras.models import load_model

# fix random seed for reproducibility
np.random.seed(7)

dataset2 = np.genfromtxt(os.path.join('data', 'full_sgd.txt'), delimiter=',', dtype=None)
data2 = dataset2[2:]

genes_test = data2[:,1]
x_test = data2[:,3:].astype(float)
x_test = x_test[:, :, np.newaxis]

model = load_model(sys.argv[1])

y_prob = np.max(model.predict(x_test), axis=1).astype(float)
y_pred = np.argmax(model.predict(x_test), axis=1)

print y_prob[0:10]


# write ypred to results.csv
with open('results_2.txt','w') as f:
    for i in xrange(len(y_pred)):
        f.write("%s\t%d\t%f\n"%(genes_test[i],y_pred[i],y_prob[i]))
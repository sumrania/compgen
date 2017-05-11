from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.layers import LSTM
from keras.optimizers import RMSprop
from keras.utils.data_utils import get_file
import keras.backend as K
import tensorflow as tf
import numpy as np
import random
import os,sys
from keras.callbacks import ModelCheckpoint

# def get_activations(model, layer, X_batch):
#     get_activations = K.function([model.layers[0].input, K.learning_phase()], [model.layers[layer].output,])
#     activations = get_activations([X_batch,0])
#     #print len(activations)
#     return activations

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
K.set_session(sess)

np.random.seed(7)
num_classes = 5

dataset1 = np.genfromtxt(os.path.join('data', 'norm_cellcycle_384_17.txt'), delimiter=',', dtype=None)
data = dataset1[1:]

# extract columns
genes_label = data[:,0]
y_all = data[:,1].astype(int)
x_all = data[:,2:-1].astype(float)

dataset2 = np.genfromtxt(os.path.join('data', 'full_sgd.txt'), delimiter=',', dtype=None)
data2 = dataset2[2:]
genes_full = data2[:,1]
#x_test = data2[:,3:].astype(float)
#x_test = x_test[:, :, np.newaxis]

tg1 = [g.lower() for g in genes_full]
tg2 = [g.lower() for g in genes_label]

#377 from full_sg
data_not_chosen = data[np.where(np.in1d(tg2, tg1))[0],0]
x_not_chosen = []
y_not_chosen = []
for i in xrange(len(tg2)):
    for j in xrange(len(tg1)):
        if tg2[i] == tg1[j]:
            x_not_chosen.append(data2[j,3:])
            y_not_chosen.append(y_all[i]-1)
x_not_chosen = np.array(x_not_chosen)
y_not_chosen = np.array(y_not_chosen)
#x_not_chosen_data = data2[np.where(np.in1d(tg1, tg2))[0],3:]

#5772 from full sgd
chosen_data = data2[np.where(np.in1d(tg1, tg2)==False)[0],3:]
# print len(chosen_data)
# print len (x_not_chosen_data)


maxlen = 3
x_train = []
y_train = []
N=17-maxlen-1 
for i in xrange(len(chosen_data)):
    for j in range(N):
        x_train.append(chosen_data[i,j: j + maxlen])
        y_train.append(chosen_data[i,j+maxlen+1])
print('nb sequences:', len(x_train))

print len(x_train)%N
# print x_train
print('Vectorization...')
X = np.array(x_train)[:,:,np.newaxis]
y = np.array(y_train)

print X.shape
# build the model: a single LSTM
print('Build model...')
model = Sequential()
model.add(LSTM(128, batch_input_shape=(N,maxlen,1), stateful=True))
model.add(Dropout(0.5))
model.add(Dense(1))

optimizer = RMSprop(lr=0.01)
model.compile(loss='mean_squared_error', optimizer=optimizer)

# train the model, output generated text after each iteration
for iteration in range(1, 2):
    print()
    print('-' * 50)
    print('Iteration', iteration)
    model.fit(X, y,
              batch_size=N,
              epochs=1,  shuffle=False)

    # serialize model to JSON
    model_json = model.to_json()
    with open("char_rnn_model.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("char_rnn_model.h5")
    print("Saved model to disk")
#load json and create model

#model.load_weights("char_rnn_model.h5")

maxlen = 3
x_test = []
y_test = []
for i in xrange(len(x_not_chosen)):
    for j in range(N):
        x_test.append(x_not_chosen[i,j: j + maxlen])
        y_test.append(x_not_chosen[i,j+maxlen+1])
print('nb sequences:', len(x_test))

print('Vectorization...')
X = np.array(x_test)[:,:,np.newaxis]
y = np.array(y_test)


loss = model.evaluate(X, y, batch_size=N, verbose=1)
print loss

def get_activations(model, layer, X_batch):
    get_activations = K.function([model.layers[0].input, K.learning_phase()], [model.layers[layer].output,])
    activations = get_activations([X_batch,0])
    return activations

print X.shape
lstm_activations = []
for i in xrange(len(x_not_chosen)):
    x_batch = X[i*N:(i+1)*N,:] 
    #all_activ = []
   # x_batch =X[i]
   # print x_batch.shape
    activ = get_activations(model, 1, x_batch)[0].flatten()
    #all_activ.append(activ)
    #lstm_activations.append(activ)
    #print len(lstm_activations)
    lstm_activations.append(activ)

print len(lstm_activations)


lstm_activations = np.array(lstm_activations)
#lstm_activations = (np.array(lstm_activations)).reshape(377,1664)

model2 = Sequential()
model2.add(Dense(832, input_shape=(1664,), activation='relu'))
model2.add(Dense(416, activation='relu'))
# model2.add(Dropout(0.5))
model2.add(Dense(num_classes, activation='softmax'))
model2.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'],class_mode='categorical')

print("Start Training...")

filepath='lstmpretrained-chckpt-{epoch:03d}-{val_acc:.2f}.hdf5'

checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')

model2.fit(lstm_activations, y_not_chosen, epochs=100, batch_size=2, validation_split = 0.1,callbacks=[checkpoint])

print("Training complete!")


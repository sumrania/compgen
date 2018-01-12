import numpy as np
import os

import keras
from keras.models import Sequential
from keras.layers import Conv1D, Dense, Dropout, MaxPooling1D, Flatten
from keras.callbacks import ModelCheckpoint

# fix random seed for reproducibility
np.random.seed(7)
num_classes = 5

dataset1 = np.genfromtxt(os.path.join('data', 'norm_cellcycle_384_17.txt'), delimiter=',', dtype=None)
data = dataset1[1:]

# extract columns
genes_train = data[:,0]
y_all = data[:,1].astype(int) - 1
x_all = data[:,2:-1].astype(float)

print y_all.shape
print np.unique(y_all).shape[0]
#y_all = keras.utils.to_categorical(y_all)

num_classes = np.unique(y_all).shape[0]

# split entire data into train set and test set
validation_split = 0.2

val_idx = np.random.choice(range(x_all.shape[0]), int(validation_split*x_all.shape[0]), replace=False)
train_idx = [x for x in range(x_all.shape[0]) if x not in val_idx]

x_train = x_all[train_idx]
y_train = y_all[train_idx]

x_train = x_train[:, :, np.newaxis]
y_train = y_train[:,np.newaxis]

x_val = x_all[val_idx]
y_val = y_all[val_idx]
#y_val = keras.utils.to_categorical(y_val)

x_val = x_val[:, :, np.newaxis]
y_val = y_val[:,np.newaxis]


print(x_train.shape[0],'train samples')
print(x_val.shape[0],'test samples')

# Create model
#number of filters for 1D conv
nb_filter = 8
filter_length = 4
window = x_train.shape[1]

model = Sequential()
model.add(Conv1D(filters=nb_filter/2,kernel_size=filter_length,activation="relu", input_shape=(window,1)))
#model.add(MaxPooling1D())
model.add(Conv1D(nb_filter=nb_filter, filter_length=filter_length, activation='relu'))
#model.add(MaxPooling1D())
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))
model.summary()
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'],class_mode='categorical')

print("Start Training...")

filepath='model-chckpt-{epoch:03d}-{val_acc:.2f}.hdf5'

checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')

model.fit(x_train, y_train, epochs=100, batch_size=2, validation_data=(x_val, y_val),callbacks=[checkpoint])

print("Training complete!")

# pred = model.predict(x_val)
# print pred
# pred = np.argmax(model.predict(x_val), axis=1)


# Print all time points for row wise max of time series 
# max_idx = np.argmax(xtrain,axis=1)
# print max_idx[:67]



	
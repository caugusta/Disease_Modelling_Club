import keras
import string
import tensorflow as tf
import numpy as np
from keras import Model
from keras.layers import Input, Dense
from keras.optimizers import SGD
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

seed = 2018
np.random.seed(seed) #this is to make sure all our results are reproducible.

#Reading in the data

with open('train_data.csv', 'r') as f: 
    train_data = f.readlines()
    
with open('test_data.csv', 'r') as f:
    test_data = f.readlines()
    
train_no_newline = [s.rstrip() for s in train_data] #remove \n from the end of each line
train_no_quotes = [s.replace('"', '') for s in train_no_newline] # remove "" from 0 at end.

#Convert the string values to integers
train_as_list = list()
for i in range(len(train_no_quotes)):
    for j in range(len(train_no_quotes[i].split())):
        train_as_list.append(int(train_no_quotes[i].split()[j]))

#Convert the list to an array
train_array = np.asarray(train_as_list)
train_array = train_array.reshape((2400, 34))

test_no_newline = [s.rstrip() for s in test_data]
test_no_quotes = [s.replace('"', '') for s in test_no_newline]
test_as_list = list()
for i in range(len(test_no_quotes)):
    for j in range(len(test_no_quotes[i].split())):
        test_as_list.append(int(test_no_quotes[i].split()[j]))
        
test_array = np.asarray(test_as_list)
test_array = test_array.reshape((600, 33))

train_no_labels = train_array[:, :-1] #means "all rows and all columns except last column"
train_labels = train_array[:,-1] #means "all rows and only last column"

#With thanks to https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/
encoder = LabelEncoder()
encoder.fit(train_labels)
Y_train = encoder.transform(train_labels)
dummy_y = np_utils.to_categorical(Y_train)

test_no_labels = test_array[:,:-1]
test_labels = test_array[:,-1]

encoder = LabelEncoder()
encoder.fit(test_labels)
Y_test = encoder.transform(test_labels)
dummy_y_test = np_utils.to_categorical(Y_test)

#Now, when I was making the datasets, I should have ensured 
#the training set had the same number of columns as the test set.
#Fixing that now....
N = test_no_labels.shape[0] #number of rows
M = test_no_labels.shape[1] #number of columns
test_no_labels_2 = np.zeros((N, M+1)) #create a matrix of 0s with N rows and M+1 columns
test_no_labels_2[:, :-1] = test_no_labels #insert test epidemics up to last column
test_no_labels = test_no_labels_2 #reassigning the variable name.
train_use = (train_no_labels - np.mean(train_no_labels))/np.std(train_no_labels)
test_use = (test_no_labels - np.mean(test_no_labels))/(np.std(test_no_labels))


#Defining the model

input_layer = Input(shape=(33,), name='input_layer') #take in an unspecified number of epidemic curves, each of length 33
hidden_1 = Dense(10, activation="sigmoid", name="hidden_1")(input_layer) # a hidden layer with 10 units that takes in the input
hidden_2 = Dense(8, activation="sigmoid", name="hidden_2")(hidden_1) #a hidden layer with 5 units that takes in compute values from the previous hidden layer
output_layer = Dense(3, activation="softmax", name="output_layer")(hidden_2) #the output layer

model = Model(inputs=[input_layer], outputs=output_layer)
model.summary()

#Training the model

num_epochs = 10 #this is the number of times we'll iterate over the training dataset.
model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
model_training = model.fit(train_no_labels, dummy_y, validation_data = (test_no_labels, dummy_y_test), epochs=num_epochs, batch_size=20, shuffle=True, verbose=2)

#With thanks to https://machinelearningmastery.com/display-deep-learning-model-training-history-in-keras/
# summarize history for accuracy
#plt.plot(model_training.history['acc'])
#plt.plot(model_training.history['val_acc'])
#plt.title('model accuracy')
#plt.ylabel('accuracy')
#plt.xlabel('epoch')
#plt.legend(['train', 'test'], loc='upper left')
#plt.show()

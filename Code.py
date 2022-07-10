#!/usr/bin/env python
# coding: utf-8
Importing Libraries and Loading Dataset
# In[1]:


import numpy as np
from sklearn import preprocessing   # sklearn preprocessing library to standardize the data.
import tensorflow as tf

# Load the data
raw_csv_data = np.loadtxt('Audiobooks_data.csv',delimiter=',')


#leaving the IDs and target column as IDs contribute nothing to our model training and targets are to be compared after we train our model
unscaled_inputs_all = raw_csv_data[:,1:-1]

# The targets are in the last column. That's how datasets are conventionally organized.
targets_all = raw_csv_data[:,-1]


# In[2]:


#Balancing the dataset

# Count how many targets are 1 (meaning that the customer did convert)
num_one_targets = int(np.sum(targets_all))
zero_targets_counter = 0

# We want to create a "balanced" dataset, so we will have to remove some input/target pairs
indices_to_remove = []
for i in range(targets_all.shape[0]):
    if targets_all[i] == 0:
        zero_targets_counter += 1
        if zero_targets_counter > num_one_targets:
            indices_to_remove.append(i)

unscaled_inputs_equal_priors = np.delete(unscaled_inputs_all, indices_to_remove, axis=0)
targets_equal_priors = np.delete(targets_all, indices_to_remove, axis=0)


# In[3]:


#Standardizing the inputs
scaled_inputs = preprocessing.scale(unscaled_inputs_equal_priors)

#Shuffling the data
shuffled_indices = np.arange(scaled_inputs.shape[0])
np.random.shuffle(shuffled_indices)

# Use the shuffled indices to shuffle the inputs and targets.
shuffled_inputs = scaled_inputs[shuffled_indices]
shuffled_targets = targets_equal_priors[shuffled_indices]

Split the dataset into train, validation and test into 80-10-10 ratio
# In[4]:


samples_count = shuffled_inputs.shape[0]


train_samples_count = int(0.8 * samples_count)
validation_samples_count = int(0.1 * samples_count)
test_samples_count = samples_count - train_samples_count - validation_samples_count

# Create variables that record the inputs and targets for training
train_inputs = shuffled_inputs[:train_samples_count]
train_targets = shuffled_targets[:train_samples_count]

# Create variables that record the inputs and targets for validation.
validation_inputs = shuffled_inputs[train_samples_count:train_samples_count+validation_samples_count]
validation_targets = shuffled_targets[train_samples_count:train_samples_count+validation_samples_count]

# Create variables that record the inputs and targets for test.
test_inputs = shuffled_inputs[train_samples_count+validation_samples_count:]
test_targets = shuffled_targets[train_samples_count+validation_samples_count:]

Model

Outline, optimizers, loss, early stopping and training
# In[5]:


input_size = 10
output_size = 2

hidden_layer_size = 50
    
# define how the model will look like
model = tf.keras.Sequential([
    # tf.keras.layers.Dense is basically implementing: output = activation(dot(input, weight) + bias)
    # it takes several arguments, but the most important ones for us are the hidden_layer_size and the activation function
    tf.keras.layers.Dense(hidden_layer_size, activation='relu'), # 1st hidden layer
    tf.keras.layers.Dense(hidden_layer_size, activation='relu'), # 2nd hidden layer
    tf.keras.layers.Dense(output_size, activation='softmax') # output layer
])


### Choose the optimizer and the loss function and the metrics we are interested in obtaining at each iteration
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])


### Training
batch_size = 100
max_epochs = 100   # epochs that we will train for (assuming early stopping doesn't kick in)

# set an early stopping mechanism
# let's set patience=2, to be a bit tolerant against random validation loss increases
early_stopping = tf.keras.callbacks.EarlyStopping(patience=2)

# fit the model
# note that this time the train, validation and test data are not iterable
model.fit(train_inputs, train_targets, batch_size=batch_size, epochs=max_epochs, 
          # callbacks are functions called by a task when a task is completed
          # task here is to check if val_loss is increasing
          callbacks=[early_stopping],
          validation_data=(validation_inputs, validation_targets),
          verbose = 2 # making sure we get enough information about the training process
          )  


# In[6]:


#Testing the Model
test_loss, test_accuracy = model.evaluate(test_inputs, test_targets)
print('\nTest loss: {0:.2f}. Test accuracy: {1:.2f}%'.format(test_loss, test_accuracy*100.))

The test accuracy is around 80% so our Model can predict the correct case 8 out of 10 times.The main idea is that if a customer has a low probability of coming back, there is no reason to spend any money on advertizing 
to him/her. If we can focus our efforts ONLY on customers that are likely to convert again, we can make great savings. 
Moreover, this model can identify the most important metrics for a customer to come back again. Identifying new customers 
creates value and growth opportunities.

So our target are those 80% the model has found and so we can improve the business.
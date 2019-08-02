import numpy as np
from keras.models import Sequential
from keras.layers import Dense

# load the dataset
dataset = np.loadtxt('./dataset/pima-indians-diabetes.csv', delimiter=',')

np.random.seed(1) # for repeatability
### shuffle the dataset (not strictly necessary)
np.random.shuffle(dataset)

# split into input (X) and output (y) variables
X = dataset[:,0:8]
Y = dataset[:,8]

## print(len(X)) ## 768 rows
## split into train and test data
train_X = X[:512]
train_Y = Y[:512]
test_X = X[512:]
test_Y = Y[512:]

# define the keras model
model = Sequential()
model.add(Dense(64, input_dim=8, activation='relu'))
model.add(Dense(24, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
# compile the keras model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# fit the keras model on the dataset
model.fit(train_X, train_Y, epochs=500, batch_size=32, verbose=2)

# evaluate the keras model
_, accuracy = model.evaluate(train_X, train_Y)
print('Train Accuracy: %.2f' % (accuracy*100))

# make class predictions with the model
predictions = model.predict_classes(test_X).reshape((256))
accuracy = np.count_nonzero( predictions == test_Y ) / len(test_Y)
print('Test Accuracy: %.2f' % (accuracy))

model_path = "./models/model.h5"
# save architecture and weights to a single file
model.save(model_path)

#######

from keras.models import load_model

# load model
model = load_model(model_path)
# summarize model (optional)
print(model.summary())


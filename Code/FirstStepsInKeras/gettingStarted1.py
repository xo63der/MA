from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import MinMaxScaler
import numpy as np

# fix random seed for reproducibility
np.random.seed(7)
scaler = MinMaxScaler(feature_range=(0,1))

# load pima indians dataset
dataset = np.loadtxt("indiabetes.csv", delimiter=",")
# split into input (X) and output (Y) variables

#?
X = scaler.fit_transform(dataset[:,0:8])
Y = dataset[:,8]

print(X.mean(axis=0))
print(X.min(axis=0))
print(X.max(axis=0))

model = Sequential()
model.add(Dense(20, input_dim=8, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Fit the model
model.fit(X, Y, validation_split=0.2, epochs=150, batch_size=10)

# evaluate the model
scores = model.evaluate(X, Y)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

predictions = model.predict(X)
# round predictions
rounded = [round(x[0]) for x in predictions]
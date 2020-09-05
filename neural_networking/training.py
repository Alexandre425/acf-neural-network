import numpy as np
from tensorflow import keras

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Make an empty list
data = [[] for x in range(12)]
# Opening and parsing the file
f = open("training_data.txt", "r")
for lin in f:
    vals = lin.replace("[", '').replace("]", '').split()    # Remove the brackets and split every space
    vals = [float(x) for x in vals]
    i = 0
    # Put each of the values in it's list
    for v in vals:
        data[i].append(v)
        i += 1
# Atributing the correct data to the inputs and outputs
X = data[0:8]
Y = data[8:12]
# Convertyng to numpy arrays
X = [np.array(x) for x in X]
Y = [np.array(x) for x in Y]

f.close()

# Neural network model
model = Sequential([
    Dense(8, activation='relu', input_shape=(8,)),
    Dense(8, activation='relu'),
    Dense(4)
])

model.compile(
    optimizer='adam',
    loss='huber_loss',
    metrics=['mean_squared_error']
)

model.fit(x=X, y=Y, batch_size=100, epochs=5)
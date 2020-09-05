import numpy as np
from tensorflow import keras

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Neural network model
model = Sequential([
    Dense(8, input_shape=(8,)),
    Dense(8, activation='relu'),
    Dense(8, activation='relu'),
    Dense(8)
])

X = [np.array(0)]*12
print(X)
print(X[0])

# Opening and parsing the file
f = open("training_data.txt", "r")
l = 0
for lin in f:
    vals = lin.replace("[", '').replace("]", '').split()    # Remove the brackets and split every space
    vals = [float(x) for x in vals]
    i = 0
    for v in vals:
        X[i] = X[i].append(v)
        i += 1
    l += 1

print(X)


f.close()

model.compile(
    optimizer='adam',
    loss='huber_loss',
    metrics=['mean_squared_error']
)
#model.fit()
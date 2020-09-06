import numpy as np
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import ballistics_gen as bgen
import json

def read_ballistics(path):
    # Make an empty list
    X = []
    Y = []
    # Opening and parsing the file
    f = open(path, "r")
    for lin in f:
        vals = lin.replace("[", '').replace("]", '').split()    # Remove the brackets and split every space
        vals = [float(x) for x in vals]
        X.append(vals[0:8])
        Y.append(vals[8:11])
    # Convertyng to numpy arrays
    X = np.array([np.array(x) for x in X])
    Y = np.array([np.array(x) for x in Y])
    f.close()

    return X, Y

ans = input("Train? (y/n) ")

if ans == "y":
    # Load existing weights and biases
    ans = input("Load? (y/n) ")
    if ans == "y":
        model = keras.models.load_model("model.h5")
    else:
        # Neural network model
        model = Sequential([
            Dense(16, activation='relu', input_shape=(8,)),
            Dense(16, activation='relu'),
            Dense(3)
        ])
        model.compile(
            optimizer='adam',
            loss='huber_loss',
            metrics=['mean_squared_error', 'mean_absolute_percentage_error']
        )

    # Generate data
    X, Y = bgen.generate_samples()
    model.fit(x=X, y=Y, batch_size=100, epochs=8)

    ans = input("Save? (y/n) ")
    if ans == "y":
        model.save('model.h5')
else:
    ans = input("Test? (y/n) ")
    if ans == "y":
        model = keras.models.load_model("model.h5")

        X, Y = bgen.generate_samples()
        model.evaluate(
            X, 
            Y
        )
        res = model.predict(X)
        for x,r,y in zip(X, res, Y):
            print(f"Inputs / prediction / actual:\n{x}\n{r*360}{y*360}\n")

ans = input("Write w's and b's? (y/n) ")
if ans == "y":
    model = keras.models.load_model("model.h5")
    vals = []
    f = open("ascii_model_sph.txt", 'w+')
    for l in model.layers:
        kp = {}
        weights, biases = l.get_weights()
        kp["shape"] = weights.shape
        kp["activation function"] = l.get_config()["activation"]
        kp["weights"] = weights.tolist()
        kp["biases"] = biases.tolist()
        vals.append(kp)
    string = json.dumps(vals)    
    f.write(string)
    f.close()
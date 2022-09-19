import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras

path = "datarobotDelay30.csv"
df = pd.read_csv(path)

feature_keys = [
    "D0",
    "D1",
    "D2",
    "D3",
    "D4",
    "D5",
    "D6",
    "D7",
    "D8",
    "D9",
    "D10",
    "D11",
    "D12",
    "D13",
    "D14",
    "D15",
    "D16",
    "D17",
    "D18",
    "D19"
]

split_fraction = 0.8
train_split = int(split_fraction * int(df.shape[0]))
step = 1
learning_rate = 0.001
batch_size = 32
epochs = 5
Exppoints=100
Age=np.arange(Exppoints)
score=np.zeros(len(Age))
score1=np.zeros(len(Age))

seed=[111, 123, 130, 240, 260, 45, 23, 46, 27, 45]
for j in range(10):
    
    for i in range(len(Age)):
        
        print(Age[i])
        future=Age[i]
        #past=1
        selected_features = [feature_keys[i] for i in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]]
        features = df[selected_features]
        features = pd.DataFrame(features.values)
  
  
        train_data = features.loc[0 : train_split - 1]
        val_data = features.loc[train_split:]    
        start = future
        end = start + train_split

        x_train = train_data[[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]].values
  
        y_train = features.iloc[start:end][[10, 11, 12, 13, 14, 15, 16, 17, 18, 19]]
  
        x_end = len(val_data) - future

        label_start = train_split + future

        x_val = val_data.iloc[:x_end][[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]].values
        y_val = features.iloc[label_start:][[10, 11, 12, 13, 14, 15, 16, 17, 18, 19]]
    
        tf.random.set_seed(seed[j])

        model= tf.keras.Sequential([
            tf.keras.layers.Dense(10, activation='relu'),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dense(10)
        ])
        model.compile(optimizer='sgd',
                 loss=tf.keras.losses.MeanSquaredError(),
                 metrics=['mse'])
        history=model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs)
        score[i]=score[i]+(np.mean(model.evaluate(x_train, y_train))-score[i])/(j+1)
        score1[i]=score1[i]+(np.mean(model.evaluate(x_val, y_val))-score1[i])/(j+1)

        np.save('TrainingErrorD30.npy',score)
        np.save('InferenceErrorD30.npy',score1)

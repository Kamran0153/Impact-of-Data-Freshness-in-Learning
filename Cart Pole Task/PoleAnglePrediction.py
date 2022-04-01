import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras

csv_path = "CartpoleData.xlsx"
df = pd.read_excel(csv_path)

feature_keys = [
    "Position",
    "Velocity",
    "Angle",
    "Angular_Velocity",
]

def normalize(data, train_split):
    data_mean = data[:train_split].mean(axis=0)
    data_std = data[:train_split].std(axis=0)
    return (data - data_mean) / data_std

Saved_Score=["TrainingLoss1.npy", "TrainingLoss5.npy", "TrainingLoss10.npy"]
Saved_TestScore=["TestingLoss1.npy", "TestingLoss5.npy", "TestingLoss10.npy"]
j=0
#number of past observations
u=[1, 5, 10]
j=0
for L in u:

  split_fraction = 0.715
  train_split = int(split_fraction * int(df.shape[0]))
  step = 1
  learning_rate = 0.001
  batch_size = 256
  epochs = 25
  Exppoints=25
  Age=np.arange(Exppoints)*4
  score=np.zeros(len(Age))
  score1=np.zeros(len(Age))
  for i in range(len(Age)):
    print(Age[i])
    past=L
    future=Age[i] 
    selected_features = [feature_keys[i] for i in [1, 2]]
    features = df[selected_features]
  
    features = normalize(features.values, train_split)
    features = pd.DataFrame(features)
  
  
    train_data = features.loc[0 : train_split - 1]
    val_data = features.loc[train_split:]    
    start = past + future
    end = start + train_split

    x_train = train_data[[0]].values
  
    y_train = features.iloc[start:end][[1]]
    sequence_length = int(past / step)
  
    dataset_train = keras.preprocessing.timeseries_dataset_from_array(
        x_train,
        y_train,
        sequence_length=sequence_length,
        sampling_rate=step,
        batch_size=batch_size,
     )
  
    x_end = len(val_data) - past - future

    label_start = train_split + past + future

    x_val = val_data.iloc[:x_end][[0]].values
    y_val = features.iloc[label_start:][[1]]

    dataset_val = keras.preprocessing.timeseries_dataset_from_array(
      x_val,
      y_val,
      sequence_length=sequence_length,
      sampling_rate=step,
      batch_size=batch_size,
     )


    for batch in dataset_train.take(1):

      inputs, targets = batch

    

      inputs = keras.layers.Input(shape=(inputs.shape[1], inputs.shape[2]))
      lstm_out = keras.layers.LSTM(64)(inputs)
      outputs = keras.layers.Dense(1)(lstm_out)

      model = keras.Model(inputs=inputs, outputs=outputs)
      model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate), loss="mse")

  

      history = model.fit(
        dataset_train,
        epochs=epochs,
        validation_data=dataset_val,
      )

    score[i]=history.history['loss'][epochs-1]
    score1[i]=history.history['val_loss'][epochs-1]

  
    np.save(Saved_Score[j],score)
    np.save(Saved_TestScore[j], score1)
    j=j+1

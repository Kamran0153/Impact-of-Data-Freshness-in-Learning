import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras

from zipfile import ZipFile
import os

uri = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/jena_climate_2009_2016.csv.zip"
zip_path = keras.utils.get_file(origin=uri, fname="jena_climate_2009_2016.csv.zip")
zip_file = ZipFile(zip_path)
zip_file.extractall()
csv_path = "jena_climate_2009_2016.csv"

df = pd.read_csv(csv_path)

"""
## Raw Data Visualization
To give us a sense of the data we are working with, each feature has been plotted below.
This shows the distinct pattern of each feature over the time period from 2009 to 2016.
It also shows where anomalies are present, which will be addressed during normalization.
"""

titles = [
    "Pressure",
    "Temperature",
    "Temperature in Kelvin",
    "Temperature (dew point)",
    "Relative Humidity",
    "Saturation vapor pressure",
    "Vapor pressure",
    "Vapor pressure deficit",
    "Specific humidity",
    "Water vapor concentration",
    "Airtight",
    "Wind speed",
    "Maximum wind speed",
    "Wind direction in degrees",
]

feature_keys = [
    "p (mbar)",
    "T (degC)",
    "Tpot (K)",
    "Tdew (degC)",
    "rh (%)",
    "VPmax (mbar)",
    "VPact (mbar)",
    "VPdef (mbar)",
    "sh (g/kg)",
    "H2OC (mmol/mol)",
    "rho (g/m**3)",
    "wv (m/s)",
    "max. wv (m/s)",
    "wd (deg)",
]

colors = [
    "blue",
    "orange",
    "green",
    "red",
    "purple",
    "brown",
    "pink",
    "gray",
    "olive",
    "cyan",
]

date_time_key = "Date Time"


def show_raw_visualization(data):
    time_data = data[date_time_key]
    fig, axes = plt.subplots(
        nrows=7, ncols=2, figsize=(15, 20), dpi=80, facecolor="w", edgecolor="k"
    )
    for i in range(len(feature_keys)):
        key = feature_keys[i]
        c = colors[i % (len(colors))]
        t_data = data[key]
        t_data.index = time_data
        t_data.head()
        ax = t_data.plot(
            ax=axes[i // 2, i % 2],
            color=c,
            title="{} - {}".format(titles[i], key),
            rot=25,
        )
        ax.legend([titles[i]])
    plt.tight_layout()

"""
This heat map shows the correlation between different features.
"""


def show_heatmap(data):
    plt.matshow(data.corr())
    plt.xticks(range(data.shape[1]), data.columns, fontsize=14, rotation=90)
    plt.gca().xaxis.tick_bottom()
    plt.yticks(range(data.shape[1]), data.columns, fontsize=14)

    cb = plt.colorbar()
    cb.ax.tick_params(labelsize=14)
    plt.title("Feature Correlation Heatmap", fontsize=14)
    plt.show()

def normalize(data, train_split):
    data_mean = data[:train_split].mean(axis=0)
    data_std = data[:train_split].std(axis=0)
    return (data - data_mean) / data_std



"""
## Data Preprocessing
Here we are picking ~300,000 data points for training. Observation is recorded every
10 mins, that means 6 times per hour. We will resample one point per hour since no
drastic change is expected within 60 minutes. We do this via the `sampling_rate`
argument in `timeseries_dataset_from_array` utility.
We are tracking data from past 720 timestamps (720/6=120 hours). This data will be
used to predict the temperature after 72 timestamps (76/6=12 hours).
Since every feature has values with
varying ranges, we do normalization to confine feature values to a range of `[0, 1]` before
training a neural network.
We do this by subtracting the mean and dividing by the standard deviation of each feature.
71.5 % of the data will be used to train the model, i.e. 300,693 rows. `split_fraction` can
be changed to alter this percentage.
The model is shown data for first 5 days i.e. 720 observations, that are sampled every
hour. The temperature after 72 (12 hours * 6 observation per hour) observation will be
used as a label.
"""

  
split_fraction = 0.715
train_split = int(split_fraction * int(df.shape[0]))
step = 6
learning_rate = 0.001
batch_size = 256
epochs = 1
Exppoints=1
Age=np.arange(Exppoints)*2+1
training_error=np.zeros(len(Age))
inference_error=np.zeros(len(Age))
for i in range(len(Age)):
  print(Age[i])
  u=24 #number of past observations
  past=6*u
  future=6*Age[i] 
  selected_features = [feature_keys[i] for i in [0, 1, 5, 7, 8, 10, 11]]
  features = df[selected_features]
  features.index = df[date_time_key]
  features.head()
  
  #print(features.std(axis=0))
  features = normalize(features.values, train_split)
  features = pd.DataFrame(features)
  features.head()
  
  train_data = features.loc[0 : train_split - 1]
  val_data = features.loc[train_split:]    
  start = past + future-1
  end = start + train_split

  x_train = train_data[[i for i in range(7)]].values
  
  y_train = features.iloc[start:end][[1]]
  sequence_length = int(past / step)
  
  dataset_train = keras.preprocessing.timeseries_dataset_from_array(
        x_train,
        y_train,
        sequence_length=sequence_length,
        sampling_rate=step,
        batch_size=batch_size,
    )
  
  x_end = len(val_data) - past - future+1

  label_start = train_split + past + future-1

  x_val = val_data.iloc[:x_end][[i for i in range(7)]].values
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
    lstm_out = keras.layers.LSTM(32)(inputs)
    outputs = keras.layers.Dense(1)(lstm_out)

    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate), loss="mse")

  

    history = model.fit(
      dataset_train,
      epochs=epochs,
     
    )
  training_error[i]=model.evaluate(dataset_train)
  inference_error[i]=model.evaluate(dataset_val)
  print(training_error[i])
  print(inference_error[i])
plt.plot(Age, training_error)
np.save('trainingerror.npy', training_error)
np.save('inferenceerror.npy', inference_error)

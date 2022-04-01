import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
import argparse

feature_keys = [
    "Action",
    "Position",
    "Dposition",
    "Velocity",
    "Dvelocity",
]

#normalization
def normalize(data, train_split):
    data_mean = data[:train_split].mean(axis=0)
    data_std = data[:train_split].std(axis=0)
    return (data - data_mean) / data_std

parser = argparse.ArgumentParser()
parser.add_argument('--Delay', type=int, required=True, help='Two Way Communication Delay')
args=parser.parse_args()

Delay=args.Delay #size of the feature
print('Two-way Delay', Delay)

if Delay==0:
  data="MountainCarData0.xlsx"
  Saved_Score="trainingErrorD0.npy"
  Saved_testScore="InferenceErrorD0.npy"
elif Delay==5:
  data="MountainCarData5.xlsx"
  Saved_Score="trainingErrorD5.npy"
  Saved_testScore="InferenceErrorD5.npy"
elif Delay==10:
  data="MountainCarData10.xlsx"
  Saved_Score="trainingErrorD10.npy"
  Saved_testScore="InferenceErrorD10.npy"


df = pd.read_excel(data)
split_fraction = 0.715
train_split = int(split_fraction * int(df.shape[0]))
Exppoints=30
Age=np.arange(Exppoints)
Train_Score=np.zeros(len(Age))
Test_Score=np.zeros(len(Age))
for i in range(len(Age)):
  L=1
  future=Age[i] 
  selected_features = [feature_keys[i] for i in [0, 1, 2, 3, 4]]
  features = df[selected_features]
  
  features = normalize(features.values, train_split)
  features = pd.DataFrame(features)
  
  
  train_data = features.loc[0 : train_split - 1]
  val_data = features.loc[train_split:]    
  start = future
  end = start + train_split

  x_train = train_data[[0, 2, 4]].values
  
  y_train = features.iloc[start:end][[1, 3]]

  x_end = len(val_data) - future

  label_start = train_split + future

  x_val = val_data.iloc[:x_end][[0, 2, 4]].values
  y_val = features.iloc[label_start:][[1, 3]]
    
  from sklearn.linear_model import LinearRegression
  regressor = LinearRegression()
  regressor.fit(x_train, y_train)
  y_pred=regressor.predict(x_train)
  y_val_pred = regressor.predict(x_val)
    
  from sklearn.metrics import mean_squared_error
  Train_Score[i]=mean_squared_error(y_train, y_pred)
  Test_Score[i]=mean_squared_error(y_val, y_val_pred)
np.save(Saved_testScore, Test_Score)
np.save(Saved_Score, Train_Score)
 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras

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

data=["MountainCarData0.xlsx", "MountainCarData1.xlsx", "MountainCarData2.xlsx", "MountainCarData3.xlsx", "MountainCarData4.xlsx", "MountainCarData5.xlsx", "MountainCarData6.xlsx", 
      "MountainCarData7.xlsx", "MountainCarData8.xlsx", "MountainCarData9.xlsx", "MountainCarData10.xlsx", "MountainCarData11.xlsx", "MountainCarData12.xlsx", "MountainCarData13.xlsx", "MountainCarData14.xlsx", "MountainCarData15.xlsx"]
Saved_Score=["trainingloss0.npy", "trainingloss1.npy", "trainingloss2.npy", "trainingloss3.npy", "trainingloss4.npy", "trainingloss5.npy", "trainingloss6.npy", "trainingloss7.npy",
             "trainingloss8.npy", "trainingloss9.npy", "trainingloss10.npy", "trainingloss11.npy", "trainingloss12.npy", "trainingloss13.npy", "trainingloss14.npy", "trainingloss15.npy"]
Saved_testScore=["testingloss0.npy", "testingloss1.npy", "testingloss2.npy", "testingloss3.npy", "testingloss4.npy", "testingloss5.npy", "testingloss6.npy", "testingloss7.npy",
             "testingloss8.npy", "testingloss9.npy", "testingloss10.npy", "testingloss11.npy", "testingloss12.npy", "testingloss13.npy", "testingloss14.npy", "testingloss15.npy"]
j=0
for d in data:
  df = pd.read_excel(d)
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
  np.save(Saved_testScore[j], Test_Score)
  np.save(Saved_Score[j], Train_Score)
  j=j+1
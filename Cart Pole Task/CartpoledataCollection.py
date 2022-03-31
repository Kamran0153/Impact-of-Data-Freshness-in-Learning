import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt

model = tf.keras.models.load_model('DQNCartpoleExp.h5')

state=env.reset()
Dataset=state
state = tf.constant(env.reset(), dtype=tf.float32)
max_steps=10000
for i in range(1, max_steps + 1):
    state = tf.expand_dims(state, 0)
    action_probs, _ = model(state)
 
    action = np.argmax(np.squeeze(action_probs))

    state, _, done, _ = env.step(action)
    if (done==True):
      state=env.reset()
    Dataset=np.vstack((Dataset,state))
    state = tf.constant(state, dtype=tf.float32)

import pandas as pd

## convert your array into a dataframe
df = pd.DataFrame(Dataset)

## save to xlsx file

filepath ='CartpoleData.xlsx'

df.to_excel(filepath, index=False)
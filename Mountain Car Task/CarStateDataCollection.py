import numpy as np
import tensorflow as tf
import gym
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt

env =gym.make('MountainCar-v0')
#To collect dataset for different communication delay, change the parameter delay
delay=10
import math

#A simple Acceleration Controller Model
def model(sp):
  if(sp[1]>0):
    ac=1
  else:
    if (sp[0]>-0.46):
      ac=0
    else:
      ac=2
  return ac

  #initial State of the car
sp=env.reset()
#initial Action of the controller
ac = model(sp)
#Ac action set at the controller, Sp0 set of state 0 of the car , Sp1 set of state 1 of the car
Ac=[]
Sp0=[]
Sp1=[]
Ac.append(ac)
Sp0.append(sp[0])
Sp1.append(sp[1])
if delay>0:

  for j in range(delay-1):
    sp, _, done, _ = env.step(ac)
    ac = model(sp)
    Ac.append(ac)
    Sp0.append(sp[0])
    Sp1.append(sp[1])
  

for i in range(10000):
  #action at the plant
  a_p=Ac[i]
  sp, _, done, _ = env.step(a_p)
  if done:
    sp=env.reset()
  #future action at the controller
  ac = model(sp)
  Ac.append(ac)
  Sp0.append(sp[0])
  Sp1.append(sp[1])

Spdelay0=Sp0[delay:]
Spdelay1=Sp1[delay:]
if delay>0:

  Ac=Ac[:-delay]
  Sp0=Sp0[:-delay]
  Sp1=Sp1[0:-delay]

Data=[]
for i in range(len(Sp0)):
  d=[Ac[i], Sp0[i], Spdelay0[i], Sp1[i], Spdelay1[i]]
  Data.append(d)

import pandas as pd

## convert your array into a dataframe
df = pd.DataFrame(Data)

## save to xlsx file

filepath ='MountainCarData15.xlsx'

df.to_excel(filepath, index=False)
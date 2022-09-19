# Impact of Data Freshness in Learning

<b>How Does Data Freshness Affect Real-time Supervised Learning?</b> \\
Md Kamran Chowdhury Shisher and Yin Sun, submitted to IEEE/ACM Transactions on Networking, September 2022. (Part of this paper is accepted in ACM MobiHoc, 2022.)

We analyze the impact of data freshness on real-time supervised learning, where a neural network is trained to infer a time-varying target (e.g., the position of the vehicle in front) based on features (e.g., video frames) observed at a sensing node (e.g., camera or lidar). 

# Motivation
In recent years, the proliferation of networked control and cyber-physical systems such as autonomous vehicle, UAV navigation, remote surgery, industrial control system has significantly boosted the need for real-time prediction. For example, an autonomous vehicle infers the trajectories of
nearby vehicles and the intention of pedestrians based on lidars and cameras installed on the vehicle. In remote surgery, the
movement of a surgical robot is predicted in real-time. These prediction problems can be solved by real-time supervised
learning, where a neural network is trained to predict a time varying target based on feature observations that are collected
from a sensing node. Due to data processing time, transmission errors, and queueing delay, the features delivered to the neural
predictor may not be fresh. The performance of networked intelligent systems depends heavily on the accuracy of realtime
prediction. Hence, it is important to understand how data freshness affects the performance of real-time supervised
learning.

To evaluate data freshness, a metric Age of information (AoI) was introduced in [[1]](https://www.youtube.com/watch?v=_z4FHuu3-ag). Let $U_t$ be the generation time of the freshest feature received by the neural predictor at time $t$. Then, the AoI of the features, as a function of time $t$, is defined as $\Delta(t) = t-U_t$, which is the time difference between the current time $t$ and the generation time $U_t$ of the freshest received feature.

One might expect that the performance of real-time supervised learning degrades monotonically as the feature becomes stale. By conducting several experiments, we show that this is not true. Experimental results show that training error and inference error can be non-monotonic functions of AoI.

# Experimental Results
<p float="left">
  <img src="/RobotInferenceError.jpg" width="200" />
  <img src="/VideoInferenceError.jpg" width="200" /> 
  <img src="/TemperatureInferenceError.jpg" width="200" />
  <img src="/WirelessCSIInferenceError.jpg" width="200" />
</p>

# Implementation

Clone the Repo:
```sh
git clone -b main https://github.com/Kamran0153/Impact-of-Data-Freshness-in-Learning.git 
cd Impact-of-Data-Freshness-in-Learning
```

Then, for different experiments, execute different python files with valid input arguments:

(a) <b>Robot state prediction in a leader-follower robotic system:</b>
The first figure illustrates the performance of robot state prediction in
a leader-follower robotic system. 
<p float="left">
  <img src="/RoboticExperimentModel.png" width="400"></p>

As illustrated in a [Youtube video](https://www.youtube.com/watch?v=_z4FHuu3-ag), the leader robot sends its state (joint angles) $X_t$ to the follower robot through a channel. One packet for updating the leader robot’s state is sent periodically to the follower robot every 20 time-slots. The transmission time of each updating packet is 20 time-slots. The follower robot moves towards the leader’s most recent state and locally controls its robotic fingers to grab an object. We constructed an environment using the Robotics System Toolbox in MATLAB. In each episode, a can is randomly generated on a table in front of the follower robot. The leader robot observes the position of the can and illustrates to the follower robot how to grab the can and place it on another table, without colliding with other objects in the environment. The rapidly-exploring random tree (RRT) algorithm is used to control the leader robot. Collision avoidance algorithm and trajectory generation algorithm are used for local control of the follower robot. The leader robot uses a neural network to predict the follower robot’s state $Y_t$. The neural network consists of one input layer, one hidden layer with 256 ReLU activation nodes, and one fully connected (dense) output layer. The dataset contains the leader and follower robots’ states in 300 episodes of continue operation. The first 80% of the dataset is used for the training and the other 20% of the dataset is used for the inference. 

```sh
python Robotic_State_Prediction/RoboticStatePrediction.py --u=1
```

(b) <b>Video Prediction:</b> 
<p float="left">
  <img src="/Video_Prediction_Model.png" width="400"></p>
  
The next figure illustrates the experimental results of supervised learning based video prediction, which are regenerated from [[2]](https://arxiv.org/abs/1804.01523). In this experiment, the video frame $V_t$ at time $t$ is predicted based on a feature $X_{t-\delta} = (V_{t-\delta}, V_{t-\delta-1})$ that is composed of two consecutive video frames, where $\Delta(t) = \delta$ is the AoI. A pre-trained neural network model called “SAVP” [[2]](https://arxiv.org/abs/1804.01523) is used to evaluate on 256 samples of “BAIR” dataset [[3]](https://arxiv.org/abs/1710.05268), which contains video frames of a randomly moving robotic arm. The pre-trained neural network model can be downloaded from the Github repository of [[2]](https://arxiv.org/abs/1804.01523).

(c) <b> Temperature Prediction Task </b>:
```sh
python Temperature_Prediction_Task/TemperaturePrediction.py --u=1
```
In the third figure, we plot the performance of temperature prediction. In this experiment, the temperature $Y_t$ at time $t$ is predicted based on a feature $X_{t-\delta} = \{s_{t-\delta}, \ldots, s_{t-\delta-u+1}\}  where $s_t$ is a 7-dimensional vector consisting of the temperature, pressure, saturation vapor pressure, vapor pressure deficit, specific humidity, airtight, and wind speed at time t. Similar to [[4]](https://keras.io/examples/timeseries/timeseries_weather_forecasting/), we have used an LSTM neural network and Jena climate dataset recorded by Max Planck Institute for Biogeochemistry. In this experiment, time unit of the sequence is 1 hour. 

(d) <b> CSI Prediction Task </b>:

```sh
python CSI_Prediction_Task/CSIPrediction.py --u=1
```
The last figure illustrates the performance of channel state information (CSI) prediction. The CSI $h_t$ at time $t$ is predicted based on a feature $X_{t-\delta} = \{h_{t-\delta}, \ldots, h_{t-\delta-u+1}\}$. The dataset for CSI is generated by using Jakes model [[5]](https://ieeexplore.ieee.org/document/1512123). 

By executing the above commands, you will get arrays of "Training Error" and "Inference Error" in '.npy' files.

To get "Training Error" and "Inference Error" for different feature size "u", you need to change the value. The value of "u" needs to be integer and greater than 1. 

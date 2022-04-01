# Impact of Data Freshness in Learning


To reproduce the figures of the paper titled "How Does Data Freshness Affect Real-time Supervised Learning", you need to follow the following steps:

git clone -b main https://github.com/Kamran0153/Impact-of-Data-Freshness-in-Learning.git

cd Impact-of-Data-Freshness-in-Learning

Then, for different experiments use:

(a) CSI Prediction Task:

python CSI_Prediction_Task/CSIPrediction.py --u=1


(b) Temperature Prediction Task:

python Temperature_Prediction_Task/TemperaturePrediction.py --u=1

(c) Pole Angle Prediction

python Cart_Pole_Task/PoleAnglePrediction.py --u=1

(d) Mountain Car State Prediction

python Mountain_Car_Task/CarStatePrediction.py --Delay=10

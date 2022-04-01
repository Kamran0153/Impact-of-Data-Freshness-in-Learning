# Impact of Data Freshness in Learning


To reproduce the figures of the paper titled "How Does Data Freshness Affect Real-time Supervised Learning?", you need to use following set of commands:

Clone the Repo:
```sh
git clone -b main https://github.com/Kamran0153/Impact-of-Data-Freshness-in-Learning.git 
cd Impact-of-Data-Freshness-in-Learning
```

Then, for different experiments, execute different python files with valid input arguments:

(a) CSI Prediction Task:

```sh
python CSI_Prediction_Task/CSIPrediction.py --u=1
```

(b) Temperature Prediction Task:
```sh
python Temperature_Prediction_Task/TemperaturePrediction.py --u=1
```
(c) Pole Angle Prediction:
```sh
python Cart_Pole_Task/PoleAnglePrediction.py --u=1
```
(d) Mountain Car State Prediction:
```sh
python Mountain_Car_Task/CarStatePrediction.py --Delay=10
```

By executing the above commands, you will get arrays of "Training Error" and "Inference Error" in '.npy' files.

To get "Training Error" and "Inference Error" for different feature size "u" and "Delay", you need to change the value. The value of "u" needs to be integer and greater than 1. The value of Delay is used only in Mountain Car State Prediction and "Delay" can be only "0, 5, 10" with the given dataset and the code. To generate figure for other values of "Delay". you need to make the dataset using "CarStateDataCollection.py" and insert the path of the dataset inside "CarStatePrediction.py".

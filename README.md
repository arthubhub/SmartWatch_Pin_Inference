# SmartWatch_Pin_Inference
This repo contains a temptative of realising an attack on a SmartWatch to infer sensitive data from the user using watch IMU.


## Objectives

The objective of this project is to infer sensitive data (pin code) from SmartWatch sensors.

## Threat model

The victim is using a SmartWatch on the wrist. The victim will have a normal behaviour, and at some time unlock its smartphone.

- We will assume that the pin code is 4 digits long.;
- We will assume that the smartwatch sensors are accessible from a malicious application without authorization (usually the case);
- We will assume that the phasis of going inside the SmartWatch and establishing a communication channel to the attackers server has been made.

## Hardware setup

To simulate the smartwatch, we are using an arduino, a gyroscope and an accelerometer.
Here is a picture of the setup:
![35855A8E-D046-40DF-A37C-0B527FF2D77D 2](https://github.com/user-attachments/assets/20c05aab-178a-4995-a6cf-c403531bf9aa)

## Data filtering

In order to extract the pin from the sensor sequences, we first nee to indentify when the victim is unlocking its smartphone.
Here is a picture of the data (smartphone unlocking sequence between other regular behaviours):
<img width="1680" height="961" alt="pin_sequence" src="https://github.com/user-attachments/assets/55d70553-e696-4e69-b52e-85d1905a16e6" />

Analysis:
- When the user holds his smartphone, the watch takes a specific orientation : Y -> 0 ; X -> [0.5;1] G ; Z -> [0;0.5] G. We can use this information to filter frames when the user is likely to hold its smartphone. By using a gaussian filter or applying a low pass filter on the accelerometer values we can get the orientation.
- When the user is typing the code, its hand stays very stable because he has to focus on th screen. We can see up to four clear peaks appearing in the graph, one per pressed digit.
- The user usually not type with the same speed and strength in regular use of his smartphone.

### Getting unlocking sequences

Here is a first idea of process to gather information about how to extract information about watch orientation:
- Analyse regular behaviour
- Analyse unlocking sequences in different contexts
- Compare the data and find caracteristics of unlocking sequences
- Chose a set of parameters to identify unlocking frames : orientation (gx,gy and accelerometer), variance of signal, mean, peaks, etc
- Try to identify unlocking sequences, compare with real data (false positive, false negatives, average performance, ...)



## Model training

Once we have extracted a sequence of unlocking smartphone, we have to train a model to deduce the pin code from sensor datas. To do so, the first phasis is to collect a bunch of pairs, sensordata/pincode. To achieve this, we have created a flask webserver that register user actions and correlate them with sensor data. Here is the scheme of the application:
<img width="680" height="734" alt="image" src="https://github.com/user-attachments/assets/fa951f83-b6cc-4b9d-9c58-796a030d207d" />

Result visualization:
<img width="1680" height="961" alt="Figure_1" src="https://github.com/user-attachments/assets/4f4474c9-d817-44e8-813e-1a5c91fe58f3" />

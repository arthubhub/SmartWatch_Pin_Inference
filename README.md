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

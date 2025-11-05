# SmartWatch_Pin_Inference
This repo contains a temptative of realising an attack on a SmartWatch to infer sensitive data from the user using watch IMU.


## Objectives

The objective of this project is to infer sensitive data (pin code) from SmartWatch sensors.

## Threat model

The victim is using a SmartWatch on the wrist. The victim will have a normal behaviour, and at some time unlock its smartphone.

- We will assume that the pin code is 4 digits long.;
- We will assume that the smartwatch sensors are accessible from a malicious application without authorization (usually the case);
- We will assume that the phasis of going inside the SmartWatch and establishing a communication channel to the attackers server has been made.

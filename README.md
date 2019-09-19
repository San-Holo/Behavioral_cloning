# Behavioral_cloning

This repository resumes my work during summer 2018. I worked on a convolutional model, inspired by Nvidia's behavioral cloning. The main idea was to use data extracted "AirSim", simulator from Microsoft, in order to train a virtual car to drive properly on a circuit that i made with Unreal Engine 4.

PyTorch was used to define the model, and create a Dataset class useful for train. The model learns to associate a picture and a speed value to a throttling factor and a sterring value.

To make the program works properly, you'll have to gather all images in a directory named "scenes_fpv", and all corresponding values in a csv file named "circuit_cw_user". ADLDataset makes it easier.

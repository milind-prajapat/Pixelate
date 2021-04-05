# Pixelate
A project based on image processing and path finding algorithm using pybullet for simulation

This project is based on the instructions given in the following PS-
[Problem Statement](https://drive.google.com/file/d/1DETWGOMy4nRsJz9X7VkrQKtws0t4Pp7y/view?usp=sharing)

## Installation Guidelines
Before downloading/cloning this repository on your local machine, it is needed to have the following repositories-
Follow the steps given in the repositories and install the requires packages

1. [Pixelate_Sample_Arena](https://github.com/Robotics-Club-IIT-BHU/Pixelate_Sample_Arena)
2. [Pixelate_Main_Arena](https://github.com/Robotics-Club-IIT-BHU/Pixelate_Main_Arena)

After that, clone this repository and now your system is ready to run the program.
You can either run the code directly on visual studio using the file Pixelate.sln 
or just run the file Solution.py.
All the details are already mentioned in the file Pixelate.py 

## Project Features
* Visual representation of the arena and the bot movements are done using pybullet.
* Image processing technique which is Open Computer Vision is used to manipulate the data( shape, color, aruco detection) in programmable form.
* The arena is converted into a 2-D matrix. Where each block of the arena is denoted by a particular node number.
* A* algorithm determines the shortest path to reach the destination. Here manhattan distance is used as the heuristic distance. The optimal path is calculated by considering weight of the particular node. Movement through one way nodes are considered by disconnecting it from the restricted nodes.
* The code also supports manual run of the bot.

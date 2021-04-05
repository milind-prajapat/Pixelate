# Pixelate
A project based on image processing and path finding algorithm using pybullet for simulation

This project is based on the instructions given in the following [Problem Statement](https://drive.google.com/file/d/1DETWGOMy4nRsJz9X7VkrQKtws0t4Pp7y/view?usp=sharing).

## Installation Guidelines
Along with this repository, it is needed to have the following repositories:
1. [Pixelate_Sample_Arena](https://github.com/Robotics-Club-IIT-BHU/Pixelate_Sample_Arena)
2. [Pixelate_Main_Arena](https://github.com/Robotics-Club-IIT-BHU/Pixelate_Main_Arena)

Follow the steps given in these repositories and install the packages required.

You can either run the code directly on visual studio using [Pixelate.sln](https://github.com/milind-prajapat/Pixelate/blob/main/Pixelate.sln) or can run [Solution.py](https://github.com/milind-prajapat/Pixelate/blob/main/Solution.py).

## Project Features
* Visual representation of the arena and the bot movements were done using pybullet.
* Image processing techniques using OpenCV was used to manipulate the data, i.e., shape, colour and aruco marker detection in programmable form.
* The arena was converted into a 2-D matrix where a particular node number denoted each square in the arena.
* A* algorithm determined the shortest path to reach the destination node. Here, manhattan distance was used as the heuristic measure. Movement through one-ways was considered by disconnecting it from the graph whenever required.
* The program also supports manual override to run the bot.

## References
[Output Video](https://drive.google.com/file/d/1H9sOwg9ko8G9HRjU_7ucStvyIivzOJvN/view?usp=sharing)

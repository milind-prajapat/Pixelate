# Pixelate
An Autonomous Bot Based On Image processing And A-Star Path Finding Algorithm

This project is based on the instructions given in the following [Problem Statement](https://drive.google.com/file/d/1XZivQZIc6szvCmp2vMksxlliCC4whAkB/view?usp=sharing).

## Installation Guidelines
Along with this repository, it is needed to have the following repositories:
1. [Pixelate_Sample_Arena](https://github.com/Robotics-Club-IIT-BHU/Pixelate_Sample_Arena)
2. [Pixelate_Main_Arena](https://github.com/Robotics-Club-IIT-BHU/Pixelate_Main_Arena)

Follow the steps given in these repositories and install the packages required.

You can then either run the code directly on the visual studio using [Pixelate.sln](https://github.com/milind-prajapat/Pixelate/blob/main/Pixelate.sln) or can run [Pixelate.py](https://github.com/milind-prajapat/Pixelate/blob/main/Solution.py).

## Approach
1. The arena was converted into a 2D matrix using image processing techniques where a particular node number denoted each square of the arena
2. A-Star Path Finding Algorithm was used to determine the shortest path to the destination node; here, we used manhattan distance as the heuristic measure
3. Movement through one-ways was considered by disconnecting it from the graph whenever required
4. We used the differential drive to run the bot more efficiently

## Features
1. Visual representation of the arena and the bot movements were done using **PyBullet**
2. **Image processing** techniques were used to manipulate the data, i.e., shape, colour and aruco marker detection in the programmable form
3. **A-Star Path Finding Algorithm** determined the shortest path to reach the destination node
4. The program also supports manual override to run the bot

## References
1. [Run-on Pixelate_Sample_Arena](https://drive.google.com/file/d/1Grus2-VQ6b7RzPIfdwOSfPk-CN6LGskj/view?usp=sharing)
2. [Run-on Pixelate_Main_Arena](https://drive.google.com/file/d/1H9sOwg9ko8G9HRjU_7ucStvyIivzOJvN/view?usp=sharing)

## Contributors
1. [Milind Prajapat](https://github.com/milind-prajapat)
2. [Megha Garg](https://github.com/tech-wiz18)
3. [Shabari S Nair](https://github.com/Killshot667)

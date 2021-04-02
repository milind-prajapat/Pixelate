import warnings
warnings.filterwarnings('ignore')

import pretty_errors
pretty_errors.configure(separator_character = '-',
                        filename_display    = pretty_errors.FILENAME_EXTENDED,
                        line_number_first   = True,
                        display_link        = True,
                        lines_before        = 5,
                        lines_after         = 2,
                        line_color          = pretty_errors.RED + '> ' + pretty_errors.default_config.line_color,
                        code_color          = '  ' + pretty_errors.default_config.line_color,
                        truncate_code       = True,
                        display_locals      = True)
pretty_errors.blacklist('c:/python')

import cv2
import gym
import math
import numpy as np
import pybullet as p
import pix_main_arena

from cv2 import aruco
from collections import Counter
from termcolor import colored

class Pixelate():

    @classmethod
    def __init__(cls, n_rows, n_cols, env_name, aruco_dict, aruco_id, write = False, filename = "output.mp4", codec = "H264", fps = 15):
        """
        initializes and computes the essential variables, interpretation_dict, color_dict, size of the arena, size of the additional area to remove and starting coordinate of the bot, also calls the Compute_Arena to compute the arena array

        Parameters
        ----------
        n_rows : int that must be greater than zero
            number of rows in the grid
        n_cols : int that must be greater than zero
            number of columns in the grid
        env_name : str
            name of the gym environment
        aruco_dict : {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20}
            dictionary of the aruco marker
        aruco_id : int
            id of the aruco marker
        write : bool, optional (the default is False, which implies not to write the frames)
            if true, writes the frames and saves them into the video, screen size is same as the size of the gym environment image
        filename : str, optional (the default is output.mp4)
            filename of the video, also include the path if you want to save the video at different location, by default it saves it in the current working directory
        codec : str must have length of exactly four, optional (the default is H264)
            codec of the video, list of supported codec can be found at: https://www.fourcc.org/codecs.php
        fps : int, optional (the default is 15)
            number of frames per second to write in the video

        Warnings
        --------
        OpenCV Exception: if codec given is incorrect, unsupported, not installed or missing, video writing will not work at all

        Raises
        ------
        TypeError
            if parameters given are not of specified type
        ValueError
            if n_rows or n_cols is zero, aruco_dict takes value other than specified values or region of interest, i.e., arena lies outside the cropped image
        error
            if tried to connect to the same rendering mode again
        """

        if not isinstance(n_rows, int):
            raise TypeError("n_rows must be an int instance")

        if not n_rows:
            raise ValueError("n_rows cannot take value zero")

        if not isinstance(n_cols, int):
            raise TypeError("n_cols must be an int instance")

        if not n_cols:
            raise ValueError("n_cols cannot take value zero")

        if not isinstance(env_name, str):
            raise TypeError("env_name must be a str instance")

        if not isinstance(aruco_dict, int):
            raise TypeError("aruco_dict must be an int instance")

        if not aruco_dict in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]:
            raise ValueError("aruco_dict cannot take value other than [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]")

        if not isinstance(aruco_id, int):
            raise TypeError("aruco_id must be an int instance")

        if not isinstance(write, bool):
            raise TypeError("write must be a bool instance")

        if not isinstance(filename, str):
            raise TypeError("filename must be a str instance")

        if not isinstance(codec, str):
            raise TypeError("codec must be a str instance")

        if not len(codec) == 4:
             raise ValueError("codec must have length of exactly four")

        if not isinstance(fps, int):
            raise TypeError("fps must be an int instance")

        cls.env = gym.make(env_name)

        cls.n_rows = n_rows
        cls.n_cols = n_cols

        cls.aruco_dict = aruco.Dictionary_get(aruco_dict)
        cls.aruco_id = aruco_id
        
        cls.interpretation_dict = {"Black": 0, "White": 1, "Green": 2, "Yellow": 3, "Red": 4, "Pink": 5, "Cyan": 7, "Blue Square": 11, "Blue Circle": 13,
                                    "Blue Triangle 0": 17, "Blue Triangle 90": 19, "Blue Triangle 180": 23, "Blue Triangle 270": 29, "Dark Green": -1}
        
        print(colored("Instructions:", "grey", "on_cyan"))
        print(colored("Crop The Image To Arena Size, press c to cancel if cropping is not required", "grey", "on_cyan"))

        cls.writer = None

        img = cls.Image()

        if write:
            cls.writer = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc(*codec), fps, (img.shape[1],img.shape[0]))

        r = cv2.selectROI(img)

        if not r == (0,0,0,0):
            crop = img[int(r[1]):int(r[1] + r[3]), int(r[0]):int(r[0] + r[2])]
            cls.size = np.array([crop.shape[1], crop.shape[0]], dtype = np.int)
            cls.thickness = np.array([r[0], r[1]], dtype = np.int)
        else:
            cls.size = np.array([img.shape[1], img.shape[0]], dtype = np.int)
            cls.thickness = np.array([0, 0], dtype = np.int)

        cls.color_dict = {}

        for color in ["White", "Green", "Yellow", "Red", "Pink", "Cyan", "Blue"]:
            print(colored(f"Select {color} Color", "grey", "on_cyan"))

            r = cv2.selectROI(img)
            crop = img[int(r[1]):int(r[1] + r[3]), int(r[0]):int(r[0] + r[2])]

            lower = np.array([crop[:,:,0].min(), crop[:,:,1].min(), crop[:,:,2].min()], dtype = np.int)
            upper = np.array([crop[:,:,0].max(), crop[:,:,1].max(), crop[:,:,2].max()], dtype = np.int)

            cls.color_dict[color] = np.array([lower, upper], np.int)

        cv2.destroyAllWindows()
        
        cls.start, _, _ = cls.Bot_Coordinates()

        cls.Compute_Arena()

    @classmethod
    def Image(cls):
        """
        captures the gym environment RGB image, also writes the frames into the video if write was true

        Returns
        -------
        numpy.ndarray of dtype int with shape same as the size of the image
            image captured from the RGB camera of the gym environment
        """

        img = cls.env.camera_feed()

        if cls.writer:
            cls.writer.write(img)

        return img

    @classmethod
    def Respawn_Bot(cls):
        """
        removes and respawns the bot at its starting coordinate
        """

        cls.env.remove_car()
        cls.env.respawn_car()

        _ = cls.Image()
    
    @classmethod
    def Reset_Environment(cls):
        """
        reset and restores the gym environment to its original state, also re-computes the arena array
        """

        cls.env.reset()
        cls.Compute_Arena()

    @classmethod
    def Grid_Coordinate(cls, coordinate):
        """
        converts the coordinate from the image coordinate system into the grid coordinate system

        Parameters
        ----------
        coordinate : numpy.ndarray of dtype int with shape (2,)
            coordinate in the image coordinate system

        Returns
        -------
        numpy.ndarray of dtype int with shape (2,)
            coordinate in the grid coordinate system

        Raises
        ------
        TypeError
            if parameters given are not of specified type
        ValueError
            if coordinate does not have a dtype int or shape (2,)
        """
        
        if not isinstance(coordinate, np.ndarray):
            raise TypeError("coordinate must be a numpy.ndarray instance")

        if not np.issubdtype(coordinate.dtype, np.integer):
            raise ValueError("coordinate must have dtype int")

        if not coordinate.shape == (2,):
            raise ValueError("coordinate must have shape (2,)")

        return np.array([(coordinate[1] - cls.thickness[1]) / (cls.size[1] / cls.n_rows),
                         (coordinate[0] - cls.thickness[0]) / (cls.size[0] / cls.n_cols)], dtype = np.int)

    @classmethod
    def Image_Coordinate(cls, coordinate):
        """
        converts the coordinate from the grid coordinate system into the image coordinate system

        Parameters
        ----------
        coordinate : numpy.ndarray of dtype int with shape (2,)
            coordinate in the grid coordinate system

        Returns
        -------
        numpy.ndarray of dtype int with shape (2,)
            coordinate in the image coordinate system

        Raises
        ------
        TypeError
            if parameters given are not of specified type
        ValueError
            if coordinate does not have a dtype int or shape (2,)
        """
        
        if not isinstance(coordinate, np.ndarray):
            raise TypeError("coordinate must be a numpy.ndarray instance")

        if not np.issubdtype(coordinate.dtype, np.int):
            raise ValueError("coordinate must have dtype int")

        if not coordinate.shape == (2,):
            raise ValueError("coordinate must have shape (2,)")

        return np.array([(coordinate[1] + 0.5) * (cls.size[0] / cls.n_cols) + cls.thickness[0],
                         (coordinate[0] + 0.5) * (cls.size[1] / cls.n_rows) + cls.thickness[1]], dtype = np.int)

    @classmethod
    def Bot_Coordinates(cls):
        """
        computes the bot coordinate in the grid coordinate system, in the image coordinate system and the bot vector in the image coordinate system

        Returns
        -------
        tuple of numpy.ndarray of dtype int with shape (2,)
            tuple of size three, containing the bot coordinate in the grid coordinate system, in the image coordinate system and the bot vector in the image coordinate system

        Raises
        ------
        RuntimeError
            if aruco with given id is not found in the cameral image
        """

        gray = cv2.cvtColor(cls.Image(), cv2.COLOR_BGR2GRAY)
        corners, ids, _ = aruco.detectMarkers(gray, cls.aruco_dict, parameters = aruco.DetectorParameters_create())
        
        for index, corner in enumerate(corners):
            id = ids[index][0]

            if id == 107:
                position = np.array([(corner[0][0][0] + corner[0][2][0]) / 2, (corner[0][0][1] + corner[0][2][1]) / 2], dtype = np.int)
                position_node = cls.Grid_Coordinate(position)             
                bot_vector = np.array([(corner[0][0][0] + corner[0][1][0] - corner[0][2][0] - corner[0][3][0]) / 2, (corner[0][0][1] + corner[0][1][1] - corner[0][2][1] - corner[0][3][1]) / 2], dtype = np.int)

                return position_node, position, bot_vector

        raise RuntimeError(f"aruco with id {cls.aruco_id} not found in the cameral image")

    @classmethod
    def Compute_Arena(cls):
        """
        initializes and computes the arena array, info_dict, also calls the Respawn_Bot to remove and respawn the bot at its starting coordinate if the bot is at different coordinate
        """
        
        bot_coordinate, _, _ = cls.Bot_Coordinates()
        if not cls.start in bot_coordinate:
            cls.Respawn_Bot()
            bot_coordinate, _, _ = cls.Bot_Coordinates()
        
        img = cls.Image()
        cls.arena = np.zeros([cls.n_rows, cls.n_cols], dtype = np.int)

        for color in ["White", "Green", "Yellow", "Red", "Pink", "Cyan", "Blue"]:
            lower, upper = cls.color_dict[color]
            mask = cv2.inRange(img, lower - 10, upper + 10)

            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
            mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel)
            res = cv2.bitwise_and(img, img, mask = mask)

            gray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
            contours, _ = cv2.findContours(gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
            
            for contour in contours:
                Area = cv2.contourArea(contour)
                if Area > 200.0:
                    m = cv2.moments(contour)
                    x = m["m10"] / m["m00"]
                    y = m["m01"] / m["m00"]
                    
                    cx, cy = cls.Grid_Coordinate(np.array([x,y], dtype = np.int))
                    _, (w, h), _ = cv2.minAreaRect(contour)
                    ratio = Area / (w * h)
                    
                    if color != "Blue":
                        if ratio > 0.9:
                            cls.arena[cx][cy] = cls.interpretation_dict[color]
                    else:
                        if ratio > 0.9:
                            cls.arena[cx][cy] *= cls.interpretation_dict["Blue Square"]  
                        elif ratio > 0.75:
                            cls.arena[cx][cy] *= cls.interpretation_dict["Blue Circle"] 
                        else:
                            x_coordinates, y_coordinates = contour.reshape(-1,2).T
                            
                            x_dict = dict(Counter(x_coordinates))
                            x_key, x_value = 0, 0
                            for key in x_dict:
                                if x_dict[key] > x_value:
                                    x_key, x_value = key, x_dict[key]

                            y_dict = dict(Counter(y_coordinates))
                            y_key, y_value = 0, 0
                            for key in y_dict:
                                if y_dict[key] > y_value:
                                    y_key, y_value = key, y_dict[key]

                            if x_value > y_value:
                                for key in x_dict:
                                    if abs(key - x_key) > 6 and key > x_key:
                                        cls.arena[cx][cy] *= cls.interpretation_dict["Blue Triangle 90"]
                                        break
                                    elif abs(key - x_key) > 6 and key < x_key:
                                        cls.arena[cx][cy] *= cls.interpretation_dict["Blue Triangle 270"]
                                        break
                            else:
                                for key in y_dict:
                                    if abs(key - y_key) > 6 and key > y_key:
                                        cls.arena[cx][cy] *= cls.interpretation_dict["Blue Triangle 180"]
                                        break
                                    elif abs(key - y_key) > 6 and key < y_key:
                                        cls.arena[cx][cy] *= cls.interpretation_dict["Blue Triangle 0"]
                                        break
        
        cls.arena[cls.start[0]][cls.start[1]] = cls.interpretation_dict["Dark Green"]
        
        cls.info_dict = {}
        cls.info_dict["Pink"] = np.array(sorted(np.array((cls.arena == cls.interpretation_dict["Pink"]).nonzero(), dtype = np.int).T, key = lambda coordinate : cls.Euclidean_Distance(coordinate, bot_coordinate)), dtype = np.int)
        cls.info_dict["Blue Square"] = np.array((cls.arena == cls.interpretation_dict["Cyan"] * cls.interpretation_dict["Blue Square"]).nonzero(), dtype = np.int).T.reshape(2)
        cls.info_dict["Blue Circle"] = np.array((cls.arena == cls.interpretation_dict["Cyan"] * cls.interpretation_dict["Blue Circle"]).nonzero(), dtype = np.int).T.reshape(2)
        cls.info_dict["Reveal"] = ["nan"] * cls.info_dict["Pink"].shape[0]

    @classmethod
    def Reveal(cls, coordinate):
        """
        removes the cover plate and reveals the shape underneath it, also calls the Update_Arena to update the arena array

        Parameters
        ----------
        coordinate : numpy.ndarray of dtype int with shape (2,)
            coordinate in the grid coordinate system, where the cover plate needs to be removed
        
        Raises
        ------
        TypeError
            if parameters given are not of specified type
        ValueError
            if coordinate does not have a dtype int or shape (2,)
        """

        if not isinstance(coordinate, np.ndarray):
            raise TypeError("coordinate must be a numpy.ndarray instance")

        if not np.issubdtype(coordinate.dtype, np.integer):
            raise ValueError("coordinate must have dtype int")

        if not coordinate.shape == (2,):
            raise ValueError("coordinate must have shape (2,)")

        cls.env.remove_cover_plate(coordinate[0], coordinate[1])
        cls.Update_Arena(coordinate)

    @classmethod
    def Update_Arena(cls, coordinate):
        """
        updates the arena array where the bot removed the cover plate, also updates the info_dict

        Parameters
        ----------
        coordinate : numpy.ndarray of dtype int with shape (2,)
            coordinate in the grid coordinate system, where the bot removed the cover plate
        
        Raises
        ------
        TypeError
            if parameters given are not of specified type
        ValueError
            if coordinate does not have a dtype int or shape (2,)
        """

        if not isinstance(coordinate, np.ndarray):
            raise TypeError("coordinate must be a numpy.ndarray instance")

        if not np.issubdtype(coordinate.dtype, np.integer):
            raise ValueError("coordinate must have dtype int")

        if not coordinate.shape == (2,):
            raise ValueError("coordinate must have shape (2,)")

        img = cls.Image()
        lower, upper = cls.color_dict["Blue"]
        mask = cv2.inRange(img, lower - 10, upper + 10)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel)
        res = cv2.bitwise_and(img, img, mask = mask)

        gray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
        contours, _ = cv2.findContours(gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        
        for contour in contours:
            Area = cv2.contourArea(contour)
            if Area > 200.0:
                m = cv2.moments(contour)
                x = m["m10"] / m["m00"]
                y = m["m01"] / m["m00"]
                    
                cx, cy = cls.Grid_Coordinate(np.array([x,y], dtype = np.int))
                
                if ([cx, cy] == coordinate).all():
                    _, (w, h), _ = cv2.minAreaRect(contour)
                    ratio = Area / (w * h)
                
                    if ratio > 0.9:
                        cls.arena[cx][cy] *= cls.interpretation_dict["Blue Square"]

                        index = np.flatnonzero((cls.info_dict["Pink"] == coordinate).all(1))[0]
                        cls.info_dict["Reveal"][index] = "Blue Square"
                    elif ratio > 0.75:
                        cls.arena[cx][cy] *= cls.interpretation_dict["Blue Circle"]

                        index = np.flatnonzero((cls.info_dict["Pink"] == coordinate).all(1))[0]
                        cls.info_dict["Reveal"][index] = "Blue Circle"
                    break

    @classmethod
    def Node(cls, coordinate):
        """
        computes the node number of the given grid coordinate

        Parameters
        ----------
        coordinate : numpy.ndarray of dtype int with shape (2,)
            coordinate in the grid coordinate system

        Returns
        -------
        int
            node number of the given coordinate

        Raises
        ------
        TypeError
            if parameters given are not of specified type
        ValueError
            if coordinate does not have a dtype int or shape (2,)
        """
        
        if not isinstance(coordinate, np.ndarray):
            raise TypeError("coordinate must be a numpy.ndarray instance")

        if not np.issubdtype(coordinate.dtype, np.integer):
            raise ValueError("coordinate must have dtype int")

        if not coordinate.shape == (2,):
            raise ValueError("coordinate must have shape (2,)")

        return int(coordinate[0] * cls.n_rows + coordinate[1])

    @classmethod
    def Coordinate(cls, node):
        """
        computes the grid coordinate of the given node number 

        Parameters
        ----------
        node : int
            node number

        Returns
        -------
        numpy.ndarray of dtype int with shape (2,)
            coordinate in the grid coordinate system
            
        Raises
        ------
        TypeError
            if parameters given are not of specified type
        """

        if not isinstance(node, int):
            raise TypeError("node must be a int instance")
        
        return np.array([node / cls.n_rows, node % cls.n_cols], np.int)

    @staticmethod
    def Manhattan_Distance(coordinate_1, coordinate_2):
        """
        computes the manhattan distance between the two points

        Parameters
        ----------
        coordinate_1 : numpy.ndarray dtype int with shape (2,)
            coordinate of the point_1
        coordinate_2 : numpy.ndarray dtype int with shape (2,)
            coordinate of the point_2

        Returns
        -------
        int
            manhattan distance between the two points

        Raises
        ------
        TypeError
            if parameters given are not of specified type
        ValueError
            if coordinate_1 or coordinate_2 does not have a dtype int or shape (2,)
        """

        if not isinstance(coordinate_1, np.ndarray):
            raise TypeError("coordinate_1 must be a numpy.ndarray instance")

        if not np.issubdtype(coordinate_1.dtype, np.int):
            raise ValueError("coordinate_1 must have dtype int")

        if not coordinate_1.shape == (2,):
            raise ValueError("coordinate_1 must have shape (2,)")

        if not isinstance(coordinate_2, np.ndarray):
            raise TypeError("coordinate_2 must be a numpy.ndarray instance")

        if not np.issubdtype(coordinate_2.dtype, np.int):
            raise ValueError("coordinate_2 must have dtype int")

        if not coordinate_2.shape == (2,):
            raise ValueError("coordinate_2 must have shape (2,)")

        return abs(coordinate_1[0] - coordinate_2[0]) + abs(coordinate_1[1] - coordinate_2[1])

    @classmethod
    def Adjacent_Nodes(cls, node):
        """
        computes the allowed adjacent nodes of the given node

        Parameters
        ----------
        node : int
            node number

        Returns
        -------
        dict
            containing allowed adjacent nodes as keys and their cost of travel as their values

        Raises
        ------
        TypeError
            if parameters given are not of specified type
        """

        if not isinstance(node, int):
            raise TypeError("node must be a int instance")
        
        node_dict = {}
        x, y = cls.Coordinate(node)
        
        for index, (x, y) in enumerate([[x + 1, y], [x, y - 1], [x - 1, y], [x, y + 1]]):
            new_node = cls.Node(np.array([x,y], np.int))

            if all([x >= 0, y >= 0, x < cls.n_rows, y < cls.n_cols]) and all([not new_node in cls.closed, cls.arena[x][y] != 0]):        
                if new_node == cls.end_node:
                    node_dict[new_node] = 0
                elif cls.arena[x][y] in [1,2,3,4]:
                    node_dict[new_node] = cls.arena[x][y]
                else:
                    Ways = [cls.arena[x][y] % cls.interpretation_dict["Blue Triangle 0"],
                            cls.arena[x][y] % cls.interpretation_dict["Blue Triangle 90"],
                            cls.arena[x][y] % cls.interpretation_dict["Blue Triangle 180"],
                            cls.arena[x][y] % cls.interpretation_dict["Blue Triangle 270"]]

                    Cost = [int(cls.arena[x][y] / cls.interpretation_dict["Blue Triangle 0"]),
                            int(cls.arena[x][y] / cls.interpretation_dict["Blue Triangle 90"]),
                            int(cls.arena[x][y] / cls.interpretation_dict["Blue Triangle 180"]),
                            int(cls.arena[x][y] / cls.interpretation_dict["Blue Triangle 270"])]
                
                    if 0 in Ways and Ways.index(0) != index:
                        node_dict[new_node] = Cost[Ways.index(0)]
                    
        return node_dict

    @classmethod
    def Path(cls, source, destination):
        """
        computes the shortest possible path from the source node to the destination node, i.e., one with the least cost
        
        Parameters
        ----------
        source : numpy.ndarray dtype int with shape (2,)
            coordinate of the source node
        destination : numpy.ndarray dtype int with shape (2,)
            coordinate of the destination node

        Raises
        ------
        TypeError
            if parameters given are not of specified type
        ValueError
            if source or destination does not have a dtype int or shape (2,)
        """

        if not isinstance(source, np.ndarray):
            raise TypeError("source must be a numpy.ndarray instance")

        if not np.issubdtype(source.dtype, np.int):
            raise ValueError("source must have dtype int")

        if not source.shape == (2,):
            raise ValueError("source must have shape (2,)")

        if not isinstance(destination, np.ndarray):
            raise TypeError("destination must be a numpy.ndarray instance")

        if not np.issubdtype(destination.dtype, np.int):
            raise ValueError("destination must have dtype int")

        if not destination.shape == (2,):
            raise ValueError("destination must have shape (2,)")

        cls.start_node = cls.Node(source)
        cls.end_node = cls.Node(destination)

        cls.table = np.zeros([cls.n_rows * cls.n_cols, 4], np.int)

        cls.open = [cls.start_node]
        cls.closed = []
        current_node = cls.start_node

        while current_node != cls.end_node:
            adjacent_nodes = cls.Adjacent_Nodes(current_node)

            for node in adjacent_nodes:
                if cls.table[node][0] == 0 or cls.table[node][0] > cls.table[current_node][0] + adjacent_nodes[node]:
                    if cls.table[node][0] == 0:
                        cls.open.append(node)

                    cls.table[node][0] = cls.table[current_node][0] + adjacent_nodes[node] 
                    cls.table[node][2] = cls.table[node][0] + cls.Manhattan_Distance(cls.Coordinate(cls.end_node), cls.Coordinate(node))
                    cls.table[node][3] = current_node

            cls.open.remove(current_node)
            cls.closed.append(current_node)
            
            best_node = cls.open[0]
            best_f_value = cls.table[cls.open[0]][2]

            for node in cls.open[1:]:
                if cls.table[node][2] < best_f_value:
                    best_node = node
                    best_f_value = cls.table[node][2]
     
            current_node = best_node

        node = cls.end_node
        Path = [cls.Coordinate(node)]
        while node != cls.start_node:
            node = int(cls.table[node][3])
            Path.append(cls.Coordinate(node))

        return np.array(Path[::-1], np.int)

    @staticmethod
    def Euclidean_Distance(coordinate_1, coordinate_2):
        """
        computes the euclidean distance between the two points

        Parameters
        ----------
        coordinate_1 : numpy.ndarray dtype int with shape (2,)
            coordinate of the point_1
        coordinate_2 : numpy.ndarray dtype int with shape (2,)
            coordinate of the point_2

        Returns
        -------
        float
            euclidean distance between the two points

        Raises
        ------
        TypeError
            if parameters given are not of specified type
        ValueError
            if coordinate_1 or coordinate_2 does not have a dtype int or shape (2,)
        """

        if not isinstance(coordinate_1, np.ndarray):
            raise TypeError("coordinate_1 must be a numpy.ndarray instance")

        if not np.issubdtype(coordinate_1.dtype, np.int):
            raise ValueError("coordinate_1 must have dtype int")

        if not coordinate_1.shape == (2,):
            raise ValueError("coordinate_1 must have shape (2,)")

        if not isinstance(coordinate_2, np.ndarray):
            raise TypeError("coordinate_2 must be a numpy.ndarray instance")

        if not np.issubdtype(coordinate_2.dtype, np.int):
            raise ValueError("coordinate_2 must have dtype int")

        if not coordinate_2.shape == (2,):
            raise ValueError("coordinate_2 must have shape (2,)")

        return math.sqrt((coordinate_1[0] - coordinate_2[0])**2 + (coordinate_1[1] - coordinate_2[1])**2)

    @staticmethod
    def Angle(vector_1, vector_2):
        """
        computes the angle between the two 2D vectors in degrees (-180 to +180)

        Parameters
        ----------
        vector_1 : numpy.ndarray of dtype int with shape (2,)
            coefficients of the vector_1
        vector_2 :numpy.ndarray of dtype int with shape (2,)
            coefficients of the vector_2

        Returns
        -------
        float
            angle between the two 2D vectors in degrees (-180 to +180)
        
        Raises
        ------
        TypeError
            if parameters given are not of specified type
        ValueError
            if vector_1 or vector_2 does not have a dtype int or shape (2,)
        """

        if not isinstance(vector_1, np.ndarray):
            raise TypeError("vector_1 must be a numpy.ndarray instance")

        if not np.issubdtype(vector_1.dtype, np.int):
            raise ValueError("vector_1 must have dtype int")

        if not vector_1.shape == (2,):
            raise ValueError("vector_1 must have shape (2,)")

        if not isinstance(vector_2, np.ndarray):
            raise TypeError("vector_2 must be a numpy.ndarray instance")

        if not np.issubdtype(vector_2.dtype, np.int):
            raise ValueError("vector_2 must have dtype int")

        if not vector_2.shape == (2,):
            raise ValueError("vector_2 must have shape (2,)")
        
        return (np.angle(complex(vector_2[0], vector_2[1]) / complex(vector_1[0], vector_1[1])) * 180) / math.pi

    @classmethod
    def Move_Bot(cls, factor, move):
        """
        moves the bot in the desired direction or aligns it, with an optimal speed

        Parameters
        ----------
        factor : float
            to choose an optimal speed, speed will depend on this factor
        move : {'F', 'B', 'L', 'R'}
            in which direction to move or align, F represents forward, B represents backward, L represents left alignment, R represents right alignment

        Raises
        ------
        TypeError
            if parameters given are not of specified type
        ValueError
            if move takes value other than specified values
        """

        if not isinstance(factor, float):
            raise TypeError("factor must be a float instance")

        if not isinstance(move, str):
            raise TypeError("move must be a str instance")

        if move not in ["F", "B", "L", "R"]:
            raise ValueError("move cannot take value other than ['F', 'B', 'L', 'R']")

        if move == "F" or move == "B":
            speed = int(min(10, max(factor - 12, 5)))

            if move == "F":
                cls.env.move_husky(speed, speed, speed, speed)
            elif move == "B":
                cls.env.move_husky(-speed, -speed, -speed, -speed)

            for _ in range(int(min(5, factor - 11))):
                p.stepSimulation()

        elif move == "L" or move == "R":
            speed = int(min(20, factor + 2))

            if move == "L":
                cls.env.move_husky(-speed, speed, -speed, speed)
            elif move == "R":
                cls.env.move_husky(speed, -speed, speed, -speed)

            for _ in range(5):
                p.stepSimulation()

        cls.env.move_husky(0, 0, 0, 0)
        p.stepSimulation()

    @classmethod
    def Follow_Path(cls, path):
        """
        makes the bot follow the given path, calls Reveal to remove the cover plate if the bot is at the node adjacent to the pink tile, also updates the info_dict if the bot is at the node adjacent to the blue square or blue circle

        Parameters
        ----------
        path : numpy.ndarray of dtype int with shape shape (,2)
            path to follow using bot, containing grid coordinate of all of the nodes including destination coordinate, excluding bot coordinate

        Raises
        ------
        TypeError
            if parameters given are not of specified type
        ValueError
            if path does not have a dtype int or shape (,2)
        """
        
        if not isinstance(path, np.ndarray):
            raise TypeError("path must be a numpy.ndarray instance")

        if not np.issubdtype(path.dtype, np.int):
            raise ValueError("path must have dtype int")

        if not path.shape[1:] == (2,):
            raise ValueError("path must have shape (,2)")

        for node in path:
            destination = cls.Image_Coordinate(node)

            while True:
                bot_coordinate, position, bot_vector = cls.Bot_Coordinates()
                
                distance = cls.Euclidean_Distance(position, destination)
                if distance > 12:
                    theta = cls.Angle(bot_vector, np.array([destination[0] - position[0], destination[1] - position[1]], dtype = np.int))
                    
                    if theta <= 5 and theta >= -5:
                        cls.Move_Bot(distance, "F")
                    elif theta < -5 and theta > -125:
                        cls.Move_Bot(-theta, "L")
                    elif theta > 5 and theta < 125:
                        cls.Move_Bot(theta, "R")
                    elif theta >= 175 or theta <= -175:
                        cls.Move_Bot(distance, "B")
                    elif theta >= 125 and theta < 175:
                        cls.Move_Bot(180 - theta, "L")
                    elif theta <= -125 and theta > -175:
                        cls.Move_Bot(180 + theta, "R") 
                else:
                    break

        for cover_plate in cls.info_dict["Pink"]:
            if cls.Euclidean_Distance(node, cover_plate) == 1.0:
                cls.Reveal(cover_plate)
        
        for index, x in enumerate(cls.info_dict["Reveal"]):
            if x != "nan" and cls.Euclidean_Distance(node, cls.info_dict[x]) == 1.0:
                cls.info_dict["Reveal"].remove(x)
                cls.info_dict["Pink"] = np.delete(cls.info_dict["Pink"], index, 0)
                    
                cls.info_dict["Pink"] = np.array(sorted(cls.info_dict["Pink"], key = lambda coordinate : cls.Euclidean_Distance(coordinate, bot_coordinate)), dtype = np.int)

        if cls.writer and Pixelate.info_dict["Pink"].shape[0] == 0:
            cls.writer.release()

    @classmethod
    def Manual_Override(cls):
        """
        allows manual override to drive the bot

        Input
        -----
        UP_ARROW
            makes the bot move in the forward direction
        DOWN_ARROW
            makes the bot move in the backward direction
        LEFT_ARROW
            makes the bot take a left turn
        RIGHT_ARROW
            makes the bot take a right turn
        c or C
            captures the gym environment RGB image
        r or R
            removes and respawns the bot at its starting coordinate
        q or Q
            quits the manual override
        """

        targetVel = 2.5
        while True:
            keys = p.getKeyboardEvents()
            for k, v in keys.items():
                if (k == p.B3G_UP_ARROW and (v & p.KEY_IS_DOWN)):
                    cls.env.move_husky(targetVel, targetVel, targetVel, targetVel)

                if (k == p.B3G_UP_ARROW and (v & p.KEY_WAS_RELEASED)):
                    cls.env.move_husky(0, 0, 0, 0)

                if (k == p.B3G_DOWN_ARROW and (v & p.KEY_IS_DOWN)):
                    cls.env.move_husky(-targetVel, -targetVel, -targetVel, -targetVel)

                if (k == p.B3G_DOWN_ARROW and (v & p.KEY_WAS_RELEASED)):
                    cls.env.move_husky(0, 0, 0, 0)

                if (k == p.B3G_LEFT_ARROW and (v & p.KEY_IS_DOWN)):
                    cls.env.move_husky(-targetVel, targetVel, -targetVel, targetVel)

                if (k == p.B3G_LEFT_ARROW and (v & p.KEY_WAS_RELEASED)):
                    cls.env.move_husky(0, 0, 0, 0)

                if (k == p.B3G_RIGHT_ARROW and (v & p.KEY_IS_DOWN)):
                    cls.env.move_husky(targetVel, -targetVel, targetVel, -targetVel)

                if (k == p.B3G_RIGHT_ARROW and (v & p.KEY_WAS_RELEASED)):
                    cls.env.move_husky(0, 0, 0, 0)

                if ((k == ord('c') or k == ord('C'))and (v & p.KEY_IS_DOWN)):
                    cls.Image()

                if ((k == ord('r') or k == ord('R'))and (v & p.KEY_IS_DOWN)):
                    cls.Respawn_Bot()

                if ((k == ord('q') or k == ord('Q'))and (v & p.KEY_IS_DOWN)):
                    return None

                p.stepSimulation()

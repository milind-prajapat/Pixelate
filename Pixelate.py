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
import pix_sample_arena
from cv2 import aruco
from collections import Counter

class Pixelate():
    @classmethod
    def __init__(cls, n_rows, n_cols, env_name, aruco_dict, aruco_id):
        """constructor

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

        Raises
        ------
        TypeError
            if parameters passed are not of specified type
        ValueError
            if n_rows or n_cols is zero or aruco_dict takes value other than specified values
        ConnectionRefusedError
            if tried to connect to same rendering mode again"""

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

        if aruco_dict not in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]:
            raise ValueError("aruco_dict cannot take value other than [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]")

        if not isinstance(aruco_id, int):
            raise TypeError("aruco_id must be an int instance")
        
        try:
            cls.env = gym.make(env_name)
        except Exception as e:
            raise ConnectionRefusedError(e)

        cls.n_rows = n_rows
        cls.n_cols = n_cols

        cls.aruco_dict = aruco.Dictionary_get(aruco_dict)
        cls.aruco_id = aruco_id
        
        cls.interpretation_dict = {"Black": 0, "White": 1, "Green": 2, "Yellow": 3, "Red": 4, "Pink": 5, "Cyan": 7, "Blue Square": 11, "Blue Circle": 13,
                                    "Blue Triangle 0": 17, "Blue Triangle 90": 19, "Blue Triangle 180": 23, "Blue Triangle 270": 29}
        
        cls.arena = np.zeros([n_rows, n_cols], dtype = np.int)
        cls.Pre_Compute()

    @classmethod
    def Image(cls):
        """captures gym environment RGB image

        Returns
        -------
        numpy.ndarray of dtype int with shape same as the size of the image
            image captured from RGB camera of gym environment"""

        return cls.env.camera_feed()

    @classmethod
    def Respawn_Bot(cls):
        """removes and respawns bot at its original coordinate"""

        cls.env.remove_car()
        cls.env.respawn_car()
        cls.Image()
    
    @classmethod
    def Grid_Coordinates(cls, coordinate):
        """converts coordinate from image coordinate system into grid coordinate system

        Parameters
        ----------
        coordinate : numpy.ndarray of dtype int with shape (2,)
            coordinate in image coordinate system

        Returns
        -------
        numpy.ndarray of dtype int with shape (2,)
            coordinate in grid coordinate system

        Raises
        ------
        TypeError
            if parameters passed are not of specified type
        ValueError
            if coordinate does not have a dtype int or shape (2,)"""
        
        if not isinstance(coordinate, np.ndarray):
            raise TypeError("coordinate must be a numpy.ndarray instance")

        if not np.issubdtype(coordinate.dtype, np.integer):
            raise ValueError("coordinate must have dtype int")

        if not coordinate.shape == (2,):
            raise ValueError("coordinate must have shape (2,)")

        return np.array([(coordinate[1] - cls.thickness[1]) / (cls.size[1] / cls.n_rows),
                         (coordinate[0] - cls.thickness[0]) / (cls.size[0] / cls.n_cols)], dtype = np.int)

    @classmethod
    def Image_Coordinates(cls, coordinate):
        """converts coordinate from grid coordinate system into image coordinate system

        Parameters
        ----------
        coordinate : numpy.ndarray of dtype int with shape (2,)
            coordinate in grid coordinate system

        Returns
        -------
        numpy.ndarray of dtype int with shape (2,)
            coordinate in image coordinate system

        Raises
        ------
        TypeError
            if parameters passed are not of specified type
        ValueError
            if coordinate does not have a dtype int or shape (2,)"""
        
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
        """calculates bot coordinate

        Returns
        -------
        tuple of numpy.ndarray of dtype int with shape (2,)
            tuple of size three containing bot coordinate in grid coordinate system, in image coordinate system and bot vector in image coordinate system
            
        Raises
        ------
        AttributeError
            if aruco with supplied id is not found in the cameral image"""

        gray = cv2.cvtColor(cls.Image(), cv2.COLOR_BGR2GRAY)
        corners, ids, _ = aruco.detectMarkers(gray, cls.aruco_dict, parameters = aruco.DetectorParameters_create())
        
        for i, corner in enumerate(corners):
            id = ids[i][0]
            if id == 107:
                position = np.array([(corner[0][0][0] + corner[0][2][0]) / 2, (corner[0][0][1] + corner[0][2][1]) / 2], dtype = np.int)
                position_node = cls.Grid_Coordinates(position)             
                bot_vector = np.array([(corner[0][0][0] + corner[0][1][0] - corner[0][2][0] - corner[0][3][0]) / 2, (corner[0][0][1] + corner[0][1][1] - corner[0][2][1] - corner[0][3][1]) / 2], dtype = np.int)

                return position_node, position, bot_vector

        raise AttributeError(f"aruco with id {cls.aruco_id} not found in the cameral image")

    @classmethod
    def Pre_Compute(cls):
        """calculates the arena array, size of the arena and size of the additional area to remove

        Raises
        ------
        IndexError
            if region of interest lies outside the cropped image"""
        
        print("Instructions:")
        print("Crop The Image To Arena Size")

        img = cls.Image()
        r = cv2.selectROI(img)
        cv2.destroyAllWindows()

        crop = img[int(r[1]):int(r[1] + r[3]), int(r[0]):int(r[0] + r[2])]
        cls.size = np.array([crop.shape[1], crop.shape[0]], dtype = np.int)
        cls.thickness = np.array([r[0], r[1]], dtype = np.int)

        for color in ["White", "Green", "Yellow", "Red", "Pink", "Cyan", "Blue"]:
            print(f"Select {color} Color")
            r = cv2.selectROI(img)
            cv2.destroyAllWindows()

            crop = img[int(r[1]):int(r[1] + r[3]), int(r[0]):int(r[0] + r[2])]
            lower = np.array([crop[:,:,0].min(), crop[:,:,1].min(), crop[:,:,2].min()], dtype = np.int) - 10
            upper = np.array([crop[:,:,0].max(), crop[:,:,1].max(), crop[:,:,2].max()], dtype = np.int) + 10
            mask = cv2.inRange(img, lower, upper)

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
                    
                    cx, cy = cls.Grid_Coordinates(np.array([x,y], dtype = np.int))
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
                                    
        (x, y), _, _ = cls.Bot_Coordinates()
        cls.arena[x][y] = cls.interpretation_dict["Green"] 
    
    @staticmethod
    def Euclidean_Distance(coordinate_1, coordinate_2):
        """calculates euclidean distance between two points

        Parameters
        ----------
        coordinate_1 : numpy.ndarray dtype int with shape (2,)
            coordinate of point_1
        coordinate_2 : numpy.ndarray dtype int with shape (2,)
            coordinate of point_2

        Returns
        -------
        float
            euclidean distance between two points

        Raises
        ------
        TypeError
            if parameters passed are not of specified type
        ValueError
            if coordinate_1 or coordinate_2 does not have a dtype int or shape (,2)"""

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
        """calculates angle between two 2D vectors in degrees (-180 to +180)

        Parameters
        ----------
        vector_1 : numpy.ndarray of dtype int with shape (2,)
            coefficients of vector_1
        vector_2 :numpy.ndarray of dtype int with shape (2,)
            coefficients of vector_2

        Returns
        -------
        float
            angle between two 2D vectors in degrees (-180 to +180)
        
        Raises
        ------
        TypeError
            if parameters passed are not of specified type
        ValueError
            if vector_1 or vector_2 does not have a dtype int or shape (,2)"""

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
        """moves the bot in the desired direction or aligns the bot with an optimal speed

        Parameters
        ----------
        factor : float
            to choose an optimal speed, speed will depend on this factor
        move : {'F', 'B', 'L', 'R'}
            in which direction to move or align, F represents forward, B represents backward, L represents left alignment, R represents right alignment

        Raises
        ------
        TypeError
            if parameters passed are not of specified type
        ValueError
            if move takes value other than specified values"""

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

            for _ in range(int(min(5, factor - 10))):
                p.stepSimulation()
        elif move == "L" or move == "R":
            speed = int(min(23, factor + 4))

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
        """makes bot follow the given path

        Parameters
        ----------
        path : numpy.ndarray of dtype int with shape shape (,2)
            path to follow using bot, containing grid coordinate of all of the nodes including destination coordinate, excluding bot coordinate

        Raises
        ------
        TypeError
            if parameters passed are not of specified type
        ValueError
            if path does not have a dtype int or shape (,2)"""
        
        if not isinstance(path, np.ndarray):
            raise TypeError("path must be a numpy.ndarray instance")

        if not np.issubdtype(path.dtype, np.int):
            raise ValueError("path must have dtype int")

        if not path.shape[1:] == (2,):
            raise ValueError("path must have shape (,2)")

        for node in path:
            destination = cls.Image_Coordinates(node)

            while True:
                _, position, bot_vector = cls.Bot_Coordinates()
                
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
        
    @classmethod
    def Manual_Override(cls):
        """allows manual override to drive the bot

        Input
        -----
        UP_ARROW
            makes bot move forward
        DOWN_ARROW
            makes bot move backward
        LEFT_ARROW
            makes bot take a left turn
        RIGHT_ARROW
            makes bot take a right turn
        c or C
            captures gym environment RGB image
        r or R
            respawns bot at its original coordinate
        q or Q
            to quit mannual override"""

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
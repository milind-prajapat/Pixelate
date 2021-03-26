from cv2 import aruco
from Pixelate import Pixelate

if __name__ == "__main__":
    Pixelate(6, 6, "pix_sample_arena-v0", aruco.DICT_ARUCO_ORIGINAL, 107)
    print(Pixelate.arena)
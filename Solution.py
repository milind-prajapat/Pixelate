import numpy as np
from Pixelate import Pixelate

if __name__ == "__main__":
    Pixelate(12, 12, "pix_sample_arena-v0", 16, 107)
    print(Pixelate.arena)
    print(Pixelate.info_dict)
    Pixelate.Follow_Path(np.array([[5,4],[5,3],[5,2],[5,1],[5,0],[4,0],[3,0],[2,0],[1,0]], np.int))
    print(Pixelate.arena)
    print(Pixelate.info_dict)
    Pixelate.Follow_Path(np.array([[0,0],[0,1],[0,2],[0,3],[0,4]], np.int))
    print(Pixelate.arena)
    print(Pixelate.info_dict)

    # Pixelate.start contains the starting coordinate of bot
    # Pixelate.arena contains the arena array
    # Pixelate.interpretation_dict contains the value corresponding to the color
    # Pixelate.info_dict contains coordinates where tile has a pink color (sorted acc to distance from bot), blue sqauare, blue circle, shape underneath pink tile
    # cover plate will be removed and Pixelate.arena, Pixelate.info will be updated automatically as soon as the bot reaches the node adjacent to pink tile, blue square or blue circle
    # 
    # 1. calulate the path from current bot coordinate to nearest pink tile, pass the path from 1,-1 (excluding the bot coordinate and the pink tile) 
    # 2. caluclate the path from the pink tile to the correct hospital, pass the path upto -1 (excluding the hospital)
    # 3. calulate the path from hospital to nearest pink tile, pass the path upto -1 (excluding the pink tile) 
    # 4. caluclate the path from the pink tile to the correct hospital, pass the path upto -1 (excluding the hospital)
    # keep in mind of the nodes to disconnect, one ways, hospital, pink tile
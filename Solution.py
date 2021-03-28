from Pixelate import Pixelate

if __name__ == "__main__":
    Pixelate(6, 6, "pix_sample_arena-v0", 16, 107)
    # Pixelate.arena contains the arena array
    # Pixelate.reveal contains coordinates where tile has a pink color
    # Pixelate.interpretation_dict contains the value corresponding to the color
    # cover plate will be removed and arena array will be updated automatically as soon as the bot reaches the node adjacent to pink tile
    # 1. calulate the path from starting coordinate to nearest pink tile, pass the path from 1,-1 (excluding the bot coordinate and the pink tile) 
    # 2. caluclate the path from the current coordinate to correct hospital, pass the path from 1, (excluding the bot coordinate)
    # Repeat step 1 and 2 for second pink tile
    # keep in mind of the nodes to disconnect, one ways, hospital, pink tile
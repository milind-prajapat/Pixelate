import numpy as np
from Pixelate import Pixelate

if __name__ == "__main__":
    Pixelate(6, 6, "pix_sample_arena-v0", 16, 107)
    Pixelate.Follow_Path(Pixelate.Path(Pixelate.start, Pixelate.info_dict["Pink"][0]))
    Pixelate.Follow_Path(Pixelate.Path(Pixelate.info_dict["Pink"][0], Pixelate.info_dict[Pixelate.info_dict["Reveal"][0]]))
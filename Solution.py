from Pixelate import Pixelate

if __name__ == "__main__":
    Pixelate(6, 6, "pix_sample_arena-v0", 16, 107)
    
    current_position = Pixelate.start
    while Pixelate.info_dict["Pink"].shape[0] != 0:
        Pixelate.Follow_Path(Pixelate.Path(current_position, Pixelate.info_dict["Pink"][0])[:-1])
        Pixelate.Follow_Path(Pixelate.Path(Pixelate.info_dict["Pink"][0], Pixelate.info_dict[Pixelate.info_dict["Reveal"][0]])[:-1])
        current_position = Pixelate.info_dict[Pixelate.info_dict["Reveal"][0]

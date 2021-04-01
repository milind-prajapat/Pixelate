from termcolor import colored
from Pixelate import Pixelate

if __name__ == "__main__":
    Pixelate(12, 12, "pix_sample_arena-v0", 16, 107, write = True)
    
    current_position = Pixelate.start

    while Pixelate.info_dict["Pink"].shape[0] != 0:

        Pixelate.Follow_Path(Pixelate.Path(current_position, Pixelate.info_dict["Pink"][0])[1:-1])
        print(colored("Picked", "grey", "on_cyan"))

        Pixelate.Follow_Path(Pixelate.info_dict["Pink"][0].reshape(1,2))

        current_position = Pixelate.info_dict[Pixelate.info_dict["Reveal"][0]]

        Pixelate.Follow_Path(Pixelate.Path(Pixelate.info_dict["Pink"][0], Pixelate.info_dict[Pixelate.info_dict["Reveal"][0]])[1:-1])
        print(colored("Dropped", "grey", "on_cyan"))

        Pixelate.Follow_Path(current_position.reshape(1,2))

        print("\n")

    print(colored("Press Enter to Exit", "grey", "on_cyan"))
    input()
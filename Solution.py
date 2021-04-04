from Pixelate import Pixelate

if __name__ == "__main__":
    Pixelate(12, 12, "pix_main_arena-v0", 16, 107)

    current_position = Pixelate.start
    max_number_of_patient = Pixelate.info_dict["Pink"].shape[0]
    
    print(f"Started : {current_position}")

    while Pixelate.info_dict["Pink"].shape[0] != 0:
        
        Path = Pixelate.Path(current_position, Pixelate.info_dict["Pink"][0])

        print(f"Going to pick patient number {max_number_of_patient - Pixelate.info_dict['Pink'].shape[0] + 1}")
        print(Path)

        Pixelate.Follow_Path(Path[1:-1])

        print(f"Patient number {max_number_of_patient - Pixelate.info_dict['Pink'].shape[0] + 1} picked")

        Pixelate.Follow_Path(Path[-1].reshape(1,2))

        current_position = Path[-1].reshape(2)
        Path = Pixelate.Path(current_position, Pixelate.info_dict[Pixelate.info_dict["Reveal"][0]])

        print(f"Going to drop patient number {max_number_of_patient - Pixelate.info_dict['Pink'].shape[0] + 1}")
        print(Path)

        Pixelate.Follow_Path(Path[1:-1])

        print(f"Patient number {max_number_of_patient - Pixelate.info_dict['Pink'].shape[0] } dropped")

        Pixelate.Follow_Path(Path[-1].reshape(1,2))

        current_position = Path[-1].reshape(2)

        print(f"Patient number {max_number_of_patient - Pixelate.info_dict['Pink'].shape[0]} completed")

    input("Press any key to exit")
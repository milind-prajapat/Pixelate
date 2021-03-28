from Pixelate import Pixelate

if __name__ == "__main__":
    Pixelate(6, 6, "pix_sample_arena-v0", 16, 107)
    print(Pixelate.arena)
    print(Pixelate.reveal)
    path = np.array([[4,5],[3,5],[2,5],[1,5],[0,5],[0,4],[0,3],[0,2],[0,1]], np.int)
    Pixelate.Follow_Path(path)
    print(Pixelate.arena)
    print(Pixelate.reveal)
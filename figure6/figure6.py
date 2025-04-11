import pandas as pd
import numpy as np

def build_node(color_array, size_array, save_path, length=160):
    assert len(size_array) == len(color_array) == length
    df_dress = pd.read_excel("data/destriux_160.xlsx")
    with open(save_path, "w") as f:
        for i in range(length):
            f.write("{} {} {} {} {} {}\n".format(
                str(df_dress["x"][i]),
                str(df_dress["y"][i]),
                str(df_dress["z"][i]),
                str(color_array[i]),
                str(size_array[i]),
                str(df_dress["node"][i]),
            ))





if __name__ == "__main__":


    example_color_array = np.load(f'T_color.npy')
    ## k_a1A
    ## A:
    example_size_array = [1 if i + 1 in [133] else 0 for i in range(160)]#SMC
    #example_size_array = [1 if i+1 in [48, 75, 76, 78, 124, 136] else 0 for i in range(160)]#EMCI
    #example_size_array = [1 if i + 1 in [3, 10, 11, 12, 14, 29, 36, 38, 39, 42, 48, 55, 74, 77, 79, 83, 86, 95, 98, 111, 144] else 0 for i in range(160)]#LMCI
    #example_size_array = [1 if i + 1 in [3, 10, 14, 17, 28, 31, 33, 35, 36, 74, 83, 84, 86, 87, 91, 98, 105, 106, 107, 149] else 0 for i in range(160)]#AD
    ## T:
    #example_size_array = [1 if i + 1 in [2, 7, 9, 11, 15, 31, 34, 36, 48, 52, 56, 59, 61, 75, 83, 87, 90, 98, 121, 123, 127, 128, 133, 144, 145, 147 ] else 0 for i in range(160)]#LMCI
    #example_size_array = [1 if i + 1 in [3, 6, 11, 14, 16, 36, 39, 44, 45, 46, 48, 52, 74, 83, 84, 86, 107, 136] else 0 for i in range(160)]#AD

    ## na2T
    ## T:
    #example_size_array = [1 if i + 1 in [16, 17, 34, 42, 53, 62, 63, 95, 98, 101, 108, 127, 147] else 0 for i in range(160)]#LMCI
    #example_size_array = [1 if i + 1 in [3, 5, 6, 7, 9, 16, 36, 39, 45, 48, 49, 74, 83, 84, 86, 107, 123, 136, 140, 145] else 0 for i in range(160)]#AD

    ## k_pTm
    ## T:
    #example_size_array = [1 if i + 1 in [5, 7, 9, 27, 28, 34, 35, 40, 42, 48, 49, 59, 61, 62, 63, 75, 81, 83, 85, 90, 95, 98, 101, 119, 120, 122, 127, 129, 133, 144, 145] else 0 for i in range(160)]#LMCI
    #example_size_array = [1 if i + 1 in [5, 6, 7, 9, 36, 39, 44, 45, 46, 48, 49, 74, 83, 84, 86, 107, 123, 136, 140, 145] else 0 for i in range(160)]#AD

    build_node(example_color_array, example_size_array, "output/SMC.node")

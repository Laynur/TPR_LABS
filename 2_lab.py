import math
import pandas as pd
import numpy as np
import load
from matplotlib import pyplot as plt


def check_sign(square, square_math):
    if (square > 0 and square_math > 0):
        print("Знак не меняется")
    if (square > 0 and square_math < 0):
            print("Знак меняется")
    if (square < 0 and square_math < 0):
        print("Знак не меняется")
    if (square < 0 and square_math > 0):
            print("Знак меняется")

k = 10


def get_graph(k,middle_First,middle_Second, m_First, m_Second, first_name, second_name):
    count = 0
    for i in range((k+2)):
        for j in range((k+2)):
            if i < (k + 1):
                if j < (k + 1):
                    if middle_First[i] >= middle_Second[j]:
                        if middle_First[i] >= middle_Second[j+1]:
                            j = j + 1
                        else:
                            count = count + 1
                    else:
                        if middle_First[i+1] <= middle_Second[j]:
                            j = j + 1
                        else:
                            count = count + 1
    print(count)
    First_Second_intervals = np.zeros((count,2))
    First_Second_frequency = np.zeros((count,1))
    print(First_Second_intervals)
    c = 0
    while c < count:
        for i in range((k + 1)):
            for j in range((k + 1)):
                for z in range(1):
                    if middle_First[i] >= middle_Second[j]:
                        if middle_First[i] >= middle_Second[j + 1]:
                            j = j + 1
                        else:
                            if middle_First[i + 1] >= middle_Second[j + 1]:
                                First_Second_intervals[c, z] = middle_First[i]
                                First_Second_intervals[c, z + 1] = middle_Second[j + 1]
                                First_Second_frequency[c, z] = m_First[i] - m_Second[j]
                                c = c + 1
                            else:
                                First_Second_intervals[c, z] = middle_First[i]
                                First_Second_intervals[c, z + 1] = middle_First[i + 1]
                                First_Second_frequency[c, z] = m_First[i] - m_Second[j]
                                c = c + 1
                    else:
                        if middle_First[i + 1] <= middle_Second[j]:
                            j = j + 1
                        else:
                            if middle_First[i+1] <= middle_Second[j + 1]:
                                First_Second_intervals[c, z] = middle_Second[j]
                                First_Second_intervals[c, z + 1] = middle_First[i + 1]
                                First_Second_frequency[c, z] = m_First[i] - m_Second[j]
                                c = c + 1
                            else:
                                First_Second_intervals[c, z] = middle_Second[j]
                                First_Second_intervals[c, z + 1] = middle_Second[j + 1]
                                First_Second_frequency[c, z] = m_First[i] - m_Second[j]
                                c = c + 1


    print(First_Second_intervals)
    print(First_Second_frequency)
    bin_dt_First_Second, bin_gr_First_Second = np.histogram(First_Second_intervals.tolist(), bins = len(First_Second_intervals.tolist()))
    Y_First_Second = First_Second_frequency
    print(bin_gr_First_Second)
    print(" ")
    print(Y_First_Second)
    for i in range(len(Y_First_Second)):
        plt.plot([bin_gr_First_Second[i], bin_gr_First_Second[i + 1]], [Y_First_Second[i], Y_First_Second[i]], color="red")
    plt.title(f"{first_name} - {second_name}")
    plt.show()

    square = First_Second_frequency[1] * (First_Second_intervals[1,1] - First_Second_intervals[1,0])
    print(square)
    square_math = square
    for i in range(2, count):
        for j in range(1):
            square_math = square_math + First_Second_frequency[i] * (First_Second_intervals[i,j+1] - First_Second_intervals[i,j])
    print(square_math)

    check_sign(square,square_math)

get_graph(k,load.middle_intervals_graph_SONY, load.middle_intervals_graph_SHARP,load.m_graph_SONY, load.m_graph_SHARP, load.sony_name, load.sharp_name)
get_graph(k,load.middle_intervals_graph_SONY, load.middle_intervals_graph_PHG, load.m_graph_SONY, load.m_graph_PHG, load.sony_name, load.phg_name)
get_graph(k,load.middle_intervals_graph_SHARP, load.middle_intervals_graph_PHG, load.m_graph_SHARP, load.m_graph_PHG, load.sharp_name, load.phg_name)
get_graph(k,load.middle_intervals_graph_PHG, load.middle_intervals_graph_SHARP, load.m_graph_PHG, load.m_graph_SHARP, load.phg_name, load.sharp_name)

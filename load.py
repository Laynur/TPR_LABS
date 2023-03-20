import math
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt


excel_SONY = pd.read_excel("C:\\data\\SONY_intervals.xlsx", index_col= 0)
middle_intervals_graph_SONY = pd.DataFrame(excel_SONY).to_numpy()
excel_SONY = pd.read_excel("C:\\data\\SONY_intervals.xlsx", index_col= 1)
m_graph_SONY = pd.DataFrame(excel_SONY).to_numpy()
print(m_graph_SONY)
print(" ")
print(middle_intervals_graph_SONY)
sony_name = "Sony"




excel_SHARP = pd.read_excel("C:\\data\\SHARP_intervals.xlsx", index_col= 0)
middle_intervals_graph_SHARP = pd.DataFrame(excel_SHARP).to_numpy()
excel_SHARP = pd.read_excel("C:\\data\\SHARP_intervals.xlsx", index_col= 1)
m_graph_SHARP = pd.DataFrame(excel_SHARP).to_numpy()
print(m_graph_SHARP)
print(" ")
print(middle_intervals_graph_SHARP)
sharp_name = "Sharp"


excel_PHG = pd.read_excel("C:\\data\\PHG_intervals.xlsx", index_col= 0)
middle_intervals_graph_PHG = pd.DataFrame(excel_PHG).to_numpy()
excel_PHG = pd.read_excel("C:\\data\\PHG_intervals.xlsx", index_col= 1)
m_graph_PHG = pd.DataFrame(excel_PHG).to_numpy()
print(m_graph_PHG)
print(" ")
print(middle_intervals_graph_PHG)
phg_name = "PHG"

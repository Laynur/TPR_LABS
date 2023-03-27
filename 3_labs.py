import array
import math
from matplotlib import pyplot as plt
import numpy as np
import openpyxl
import pandas as pd



def compare(A1,A2,A3):
    if A1 > A2 and A1 > A3:
        print("A1 предпочтительнее A2 и A3")
    elif A2 > A1 and A2 > A3:
        print("A2 предпочтительнее A1 и A3")
    else:
        print("A3 предпочтительнее A1 и A2")

def get_graph(x,y):
    plt.scatter(x, y, color='red')
    plt.title('Эмпирическая функция полезности денег u(x)')
    plt.xlabel('x')
    plt.ylabel('u(x)')
    plt.show()

def creat_mid_inter(data,k):
    xmin = min(data)
    xmax = max(data)
    N = len(data)
    h = (xmax - xmin) / k
    start_interval = [i for i in range(0, k)]
    end_interval = [i for i in range(0, k)]

    for i in range(k):
        start_interval[i] = xmin
        end_interval[i] = start_interval[i] + h
        xmin = end_interval[i]
    middle_interval = [0 for i in range(0, k)]
    print("\nMiddle Interval при k =",f"{k} :")
    for i in range(k):
        middle_interval[i] = (start_interval[i] + end_interval[i]) / 2
    frequency = [0 for i in range(0, k)]
    j = 0
    for i in range(k):
        count = 0
        for j in range(0, N):
            if (start_interval[i] <= data[j] <= end_interval[i]):
                count += 1
        frequency[i] = count
    relative_freq = [0 for i in range(0, k)]
    for i in range(k):
        relative_freq[i] = round(frequency[i] / N, 5)
    return middle_interval, relative_freq


def u(x, n):
    # if (x < 0):
    #     ux = (-1) * abs(x**(1/(n+2)))
    # else:
    #     ux = x**(1/(n+2))
    if x >= 0:
        ux = (-1) * abs(x ** (1 / (n + 2)))
    else:
        ux = (-1) * abs(abs(x) ** (1 / (n + 2)))
    return ux

def u_my(x, X_val, U):
    return np.interp(x, X_val, U)
def get_Vux(x,p,k,n):
    ux = [u(x[i][0].value, n) for i in range(k)]
    VuX = sum([ux[i] * p[i][0].value for i in range(k)])
    return VuX,ux

def get_Vux_all(x,p,k,n):
    ux = [u(x[i], n) for i in range(k)]
    VuX = sum([ux[i] * p[i] for i in range(k)])
    return VuX, ux

def get_my_Vux(x,p,k,n):
    ux = [u_my(x[i], n) for i in range(k)]
    VuX = sum([ux[i] * p[i] for i in range(k)])
    return VuX, ux

def punkt_4(data):
    result = []
    for my_k in range(5,16):
        x_t, p_t = creat_mid_inter(data, my_k)
        x_list = np.array(x_t).tolist()
        x_list = sum(x_list, [])
        print(x_list, p_t)
        VuX_res, ux_res = get_Vux_all(x_list, p_t, my_k, n)
        result.append([VuX_res,ux_res])
    print("\n")
    print(result)
book = openpyxl.open('lr3.xlsx')
sheet=book.active


x1 = sheet['A2:A11']
p1 = sheet['B2:B11']
x2 = sheet['C2:C11']
p2 = sheet['D2:D11']
x3 = sheet['E2:E11']
p3 = sheet['F2:F11']


k = 10
n = 7
#1 - пункт
VuX1, ux1 = get_Vux(x1,p1,k,n)
VuX2, ux2 = get_Vux(x2,p2,k,n)
VuX3, ux3 = get_Vux(x3,p3,k,n)
print(VuX1, VuX2, VuX3)
compare(VuX1, VuX2, VuX3)

#2 - пункт
x_values1 = [x1[i][0].value for i in range(k) ]
x_values2 = [x2[i][0].value for i in range(k) ]
x_values3 = [x3[i][0].value for i in range(k) ]

y_values1 = [u_my(x, x_values1, ux1) for x in x_values1]
y_values2 = [u_my(x, x_values2, ux2) for x in x_values2]
y_values3 = [u_my(x, x_values3, ux3) for x in x_values3]

print(y_values1)
print(y_values2)
print(y_values3)

plt.grid(True)
# Строим график функции u(x)

# get_graph(x_values1, y_values1)
# get_graph(x_values2, y_values2)
# get_graph(x_values3, y_values3)

#3 - пункт
V1_my=sum([y_values1[i]*x_values1[i] for i in range(k)])
V2_my=sum([y_values2[i]*x_values2[i] for i in range(k)])
V3_my=sum([y_values3[i]*x_values3[i] for i in range(k)])

print(V1_my,V2_my,V3_my)
compare(V1_my,V2_my,V3_my)

# 4 - пункт
#k = 5 .. 15

# my_k = 5
# x5,p5 = creat_mid_inter(data,my_k)
# x5_list = np.array(x5).tolist()
# x5_list = sum(x5_list,[])
# print(x5_list,p5)
# VuX5, ux5 = get_Vux_all(x5_list,p5,my_k,n)
excel = pd.read_excel('C:\\AllPycharn\\TRP\\data\\SONY.xlsx')
data = pd.DataFrame(excel).to_numpy()
excel2 = pd.read_excel('C:\\AllPycharn\\TRP\\data\\SHARP.xlsx')
data2 = pd.DataFrame(excel2).to_numpy()

excel3 = pd.read_excel('C:\\AllPycharn\\TRP\\data\\PHG.xlsx')
data3 = pd.DataFrame(excel3).to_numpy()

punkt_4(data)
punkt_4(data2)
punkt_4(data3)


import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def getK(n):
    k = 1 + 3.322 * math.log10(n)
    return round(k)
def VaRalp(alpha, invert_relative_freq, invert_middle_interval):
    invert_relative = invert_relative_freq[0]
    for i in range(k):
        if invert_relative <= alpha:
            invert_relative = invert_relative + invert_relative_freq[i + 1]
            indx = i + 1
    VaR = invert_middle_interval[indx]
    print(VaR)
    return indx
def CVaRalp(index, invert_middle_interval, invert_relative_freq):
    CVaR = 0
    invrel = 0
    for i in range(index,k):
        CVaR = CVaR + (invert_middle_interval[i] * invert_relative_freq[i])
        invrel = invrel + invert_relative_freq[i]
    CVaR = CVaR/invrel
    print(CVaR)
excel = pd.read_excel('C:\\AllPycharn\\TRP\\data\\SONY.xlsx')
data = pd.DataFrame(excel).to_numpy()
print('Исходные данные: \n', pd.DataFrame(data))

N = len(data)

print("N:",N)
xmin = min(data)
xmax = max(data)
k = getK(N)
h = (xmax - xmin)/k
print("xmin:",xmin,"\nxmax:",xmax)
#k = getK(N)
print("k:", k)
print("h:", h)
start_interval = [i for i in range(0, k)]
end_interval = [i for i in range(0,k)]

for i in range(k):
    start_interval[i] = xmin
    end_interval[i] = start_interval[i] + h
    xmin = end_interval[i]

print("Interval:")
print("Start\t","\t\tEnd")
for i in range(k):
    print(start_interval[i],"\t\t",end_interval[i])
middle_interval = [i for i in range(0,k)]
print("\nMiddle Interval:")
for i in range(k):
    middle_interval[i] = (start_interval[i]+end_interval[i])/2
    print(middle_interval[i])
count = 0

#частота
frequency = [0 for i in range(0,k)]
j=0
for i in range(k):
    count = 0
    # while(start_interval[i]<=data[j]<=end_interval[i]):
    #     count +=1
    #     j+=1
    for j in range(0,N):
        if (start_interval[i] <= data[j] <= end_interval[i]):
            count+=1
    frequency[i] = count
print("\nЧастота:")
print(frequency)
#относительная частота
print("\nОтносительная частота:")
relative_freq = [0 for i in range(0,k)]
for i in range(k):
    relative_freq[i] = round(frequency[i]/N,5)
print(relative_freq)
#накопительная частота
print("\nНакопительная частота:")
cumulative_freq = [0 for i in range(0,k)]
cumulative_freq[0] = frequency[0]
for i in range(1,k):
    cumulative_freq[i] = cumulative_freq[i-1] + frequency[i]
print(cumulative_freq)

#Мат.ожидание
print("\nМат.ожидание:")
math_expect = 0
for i in range(k):
    math_expect = math_expect + (middle_interval[i] * relative_freq[i])
print(math_expect)

#Дисперсия
print("\nДисперсия:")
dispers = 0
matsq = 0
for i in range(k):
    matsq = matsq + ((middle_interval[i])**2 * relative_freq[i])
dispers = matsq - math_expect**2
print(dispers)
#Стандартное отклонение
print("\nСтандартное отклонение:")
standard_deviation =0
standard_deviation = math.sqrt(dispers)
print(standard_deviation)
#Левосторонний момент n-го порядка
n = 2
print("\nЛевосторонний момент n-го порядка:")
moment = [0 for i in range(0,k)]
left_moment = 0
for i in range(k):
    if max(math_expect - middle_interval[i],0)!=0:
        moment[i] = math_expect - middle_interval[i]
for i in range(k):
    left_moment = left_moment + (moment[i]**n)*relative_freq[i]
print(left_moment)
#Стандартный левосторонний момент n-го порядка
print("\nСтандартный левосторонний момент n-го порядка:")
stand_moment = pow(left_moment,1/n)
print(stand_moment)


invert_middle_interval = [i for i in range(0,k)]
invert_relative_freq = [i for i in range(0,k)]
for i in range(k):
    invert_middle_interval[i] = -1 * middle_interval[i]
    invert_relative_freq[i] = relative_freq[i]
invert_middle_interval.reverse()
invert_relative_freq.reverse()

print("Инверт. Middle Interval:",invert_middle_interval)
print("Инверт. Относительная частота:",invert_relative_freq)

alpha1 = 0.95
alpha2 = 0.97
alpha3 = 0.99

print("\nVaR при alph = 0.95:")
index1 = VaRalp(alpha1,invert_relative_freq,invert_middle_interval) + 1
print("\nCVaR при alph = 0.95:")
CVaRalp(index1,invert_middle_interval,invert_relative_freq)


print("\nVaR при alph = 0.97:")
index2 = VaRalp(alpha2,invert_relative_freq,invert_middle_interval) + 1
print("\nCVaR при alph = 0.97:")
CVaRalp(index2,invert_middle_interval,invert_relative_freq)


print("\nVaR при alph = 0.99:")
index3 = VaRalp(alpha3,invert_relative_freq,invert_middle_interval) + 1
print("\nCVaR при alph = 0.99:")
CVaRalp(index3,invert_middle_interval,invert_relative_freq)



bin_dt, bin_gr = np.histogram(middle_interval, bins=len(middle_interval))

Y = np.cumsum(relative_freq)
for i in range(len(Y)):
    plt.plot([bin_gr[i], bin_gr[i + 1]], [Y[i], Y[i]], color='green')
plt.show()

df = pd.DataFrame(middle_interval,Y)
df.to_excel(excel_writer="C:\\AllPycharn\\TRP\\data\\middleSony.xlsx")
